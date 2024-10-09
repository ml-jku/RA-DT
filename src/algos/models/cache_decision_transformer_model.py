import copy
import torch
import math
import torch.nn as nn
from transformers.models.decision_transformer.modeling_decision_transformer import Conv1D
from .online_decision_transformer_model import OnlineDecisionTransformerModel
from .discrete_decision_transformer_model import DiscreteDTModel


class CacheDTModel(OnlineDecisionTransformerModel):
    
    def __init__(
        self,
        config, 
        observation_space, 
        action_space,
        separate_ca_embed=True,
        detach_ca_embed=False,
        rtg_condition_ca=True,
        crossattn_encoder_layers=None,
        **kwargs
    ):
        super().__init__(config, observation_space, action_space, **kwargs)
        self.separate_ca_embed = separate_ca_embed
        self.detach_ca_embed = detach_ca_embed
        self.rtg_condition_ca = rtg_condition_ca
        self.crossattn_encoder_layers = crossattn_encoder_layers
        self.crossattn_encoder = None         
        if self.separate_ca_embed: 
            # deepcopy original embeddings --> no need to redo setting up for DiscreteDT
            self.crossattn_embed_timestep = copy.deepcopy(self.embed_timestep)
            self.crossattn_embed_state = copy.deepcopy(self.embed_state)
            self.crossattn_embed_action = copy.deepcopy(self.embed_action)
            self.crossattn_embed_ln = copy.deepcopy(self.embed_ln)
            if self.rtg_condition: 
                self.crossattn_embed_return = copy.deepcopy(self.embed_return)
            if self.reward_condition: 
                self.crossattn_embed_reward = copy.deepcopy(self.embed_rewards)
            if self.global_pos_embds: 
                self.crossattn_wpe = copy.deepcopy(self.encoder.wpe)
            if self.crossattn_encoder_layers is not None: 
                # simple solution: copy first n layers of encoder
                self.crossattn_encoder = copy.deepcopy(self.encoder.h[:self.crossattn_encoder_layers])
                self.crossattn_encoder_ln = copy.deepcopy(self.encoder.ln_f)
                # remove crossattn_layers 
                for i in range(self.crossattn_encoder_layers): 
                    if hasattr(self.crossattn_encoder[i], "crossattention"): 
                        del self.crossattn_encoder[i].crossattention
                    if hasattr(self.crossattn_encoder[i], "ln_cross_attn"):
                        del self.crossattn_encoder[i].ln_cross_attn
                    self.crossattn_encoder[i].is_cross_attention = False
            self.post_init()
        
    def compute_hidden_states(
            self,
            states=None,
            actions=None,
            rewards=None,
            returns_to_go=None,
            timesteps=None,
            attention_mask=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None,
            prompt=None,
            task_id=None, 
            context_trjs=None,
            inference_params=None,
            past_key_values=None,
            use_inference_cache=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # prepare retrieved context trjs as cross attention inputs
        crossattn_infos, crossattn_inputs, crossattn_mask = self.compute_crossattn_inputs(
            context_trjs
        )
        
        batch_size, seq_length = actions.shape[0], actions.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        inputs, stacked_inputs, stacked_attention_mask = self.compute_inputs(
            states, actions, returns_to_go, rewards, timesteps, attention_mask,
            use_inference_cache=use_inference_cache and past_key_values is not None,
        )
        
        # make position ids
        if self.global_pos_embds:
            position_ids = torch.arange(stacked_attention_mask.shape[1],
                                        device=stacked_inputs.device, dtype=torch.long).unsqueeze(0)
        else: 
            position_ids = torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long)

        if use_inference_cache and past_key_values is not None: 
            # keep only tokens of last step, as rest is cached in past_key_values
            num_tokens = max([pos for tokpos in self.tok_to_pos.values() 
                              for pos in ([tokpos] if isinstance(tokpos, int) else list(tokpos))]) + 1
            stacked_inputs = stacked_inputs[:, -num_tokens:]
            position_ids = position_ids[:, -num_tokens:]
            # remove very first step from past_key_values
            # contains: contains n_layer tuples, each tuple has 2 tensorsof shape (bs, heads, seq_len, head_dim)
            past_key_values = [tuple([past[0][:, :, num_tokens:], past[1][:, :, num_tokens:]]) 
                                                 for past in past_key_values]
            seq_length = 1
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # exploits default behaviour of DecisionTransformerGPT2Block to add cross attention on retrieved context
            encoder_hidden_states=crossattn_inputs,
            encoder_attention_mask=crossattn_mask,
            use_cache=use_inference_cache,
            past_key_values=past_key_values
        )
        # grab last hidden state
        x = encoder_outputs['last_hidden_state']

        if prompt is not None and not self.config.add_cross_attention:
            x = x[:, -seq_length * len(inputs):]
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, len(inputs), self.hidden_size).permute(0, 2, 1, 3)
        # [batch_size, r_s_a, seq_len, hidden_size]
        return x, encoder_outputs, crossattn_infos
    
    def construct_inputs_and_masks(self, state_embeddings, action_embeddings, returns_embeddings,
                                   rewards_embeddings, attention_mask, time_embeddings=None, 
                                   reward_condition=False, rtg_condition=True, action_condition=True):
        inputs, masks =  super().construct_inputs_and_masks(
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings, 
            attention_mask, time_embeddings, reward_condition, rtg_condition, action_condition
        )
        return inputs, masks
    
    def compute_crossattn_inputs(self, context_trjs):
        crossattn_inputs, crossattn_mask = None, None
        if context_trjs is not None: 
            if hasattr(self.config, "chunked") and self.config.chunked is not None: 
                # provide number of chunks + tokens per timestep to model for chunked CA computation
                self.config.n_chunks = context_trjs["states"].shape[1]
                self.config.tok_per_step = max([
                    pos for tokpos in self.tok_to_pos.values() 
                    for pos in ([tokpos] if isinstance(tokpos, int) else list(tokpos))
                ]) + 1
                # reshape from [batch_size, num_chunks, cache_len + cache_len_future, ...] to
                # [batch_size, num_chunks * (cache_len + cache_len_future), ...]
                # reshape such that chunks are embedded and position information is added correctly
                batch_size = context_trjs["states"].shape[0]
                context_trjs = {k: v.flatten(start_dim=0, end_dim=1) for k, v in context_trjs.items()}
                
            states, actions, returns_to_go, rewards, timesteps, attention_mask = context_trjs["states"], \
                context_trjs["actions"], context_trjs["returns_to_go"], context_trjs["rewards"], \
                context_trjs["timesteps"], context_trjs["attention_mask"]
            
            # move to device
            states, actions, returns_to_go, rewards, timesteps, attention_mask = states.to(self.device), \
                actions.to(self.device), returns_to_go.to(self.device), rewards.to(self.device), \
                timesteps.to(self.device), attention_mask.to(self.device)
            
            inputs_fn = self._compute_cross_attn_inputs if self.separate_ca_embed else self.compute_inputs
            _, crossattn_inputs, crossattn_mask = inputs_fn(
                states, actions, returns_to_go, rewards, timesteps, attention_mask
            )
            # if rope is True, no need to add global positions here. happens withing DT
            rope = self.config.rope if hasattr(self.config, "rope") else False
            if self.global_pos_embds and not rope:
                position_ids = torch.arange(crossattn_mask.shape[1],
                                            device=crossattn_mask.device, dtype=torch.long).unsqueeze(0)
                if self.separate_ca_embed: 
                    crossattn_inputs = crossattn_inputs + self.crossattn_wpe(position_ids)
                else: 
                    crossattn_inputs = crossattn_inputs + self.encoder.wpe(position_ids)
            
            if self.detach_ca_embed: 
                crossattn_inputs, crossattn_mask = crossattn_inputs.detach(), crossattn_mask.detach()

            if hasattr(self.config, "chunked") and self.config.chunked is not None: 
                crossattn_inputs = crossattn_inputs.reshape(batch_size, -1, self.config.hidden_size)
                crossattn_mask = crossattn_mask.reshape(batch_size, -1)
                
            if self.crossattn_encoder is not None: 
                # iterate encoder blocks --> DecisionTransformerGPT2Block forward()
                crossattn_encoder_mask = (1.0 - crossattn_mask[:, None, None, :]) * torch.finfo(torch.float16).min
                for block in self.crossattn_encoder:
                    crossattn_inputs = block(
                        hidden_states=crossattn_inputs,
                        attention_mask=crossattn_encoder_mask,
                    )[0]
                crossattn_inputs = self.crossattn_encoder_ln(crossattn_inputs)
                
        return {}, crossattn_inputs, crossattn_mask
    
    def _compute_cross_attn_inputs(self, states, actions, returns_to_go, rewards, timesteps,
                                   attention_mask): 
        """Only difference to compute_inputs() is that the cross_attn embeddings are used here."""        
        batch_size, seq_length = actions.shape[0], actions.shape[1]

        # embed each modality with a different head
        state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings = self.embed_crossattn_inputs(
            states, actions, returns_to_go, rewards, attention_mask
        )

        if self.use_time_embds:
            time_embeddings = self.get_crossattn_time_embeddings(timesteps, attention_mask=attention_mask)
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings = self.add_pos_embeddings(
                time_embeddings, state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings
            )
        else:
            time_embeddings = None

        # prepare inputs + masks
        inputs, masks = self.construct_inputs_and_masks(
            state_embeddings, action_embeddings, returns_embeddings, rewards_embeddings,
            attention_mask, time_embeddings=time_embeddings,  
            reward_condition=self.reward_condition, rtg_condition=self.rtg_condition_ca, 
            action_condition=self.action_condition
        )
        stacked_inputs, stacked_attention_mask = self.prepare_inputs_and_masks(inputs, masks, 
                                                                               batch_size, seq_length, 
                                                                               ln=self.crossattn_embed_ln)        
        return inputs, stacked_inputs, stacked_attention_mask    
    
    def embed_crossattn_inputs(self, states, actions, returns_to_go, rewards, attention_mask):
        if len(states.shape) > 4:
            # is_image_space
            states = states.float() / 255.0
        state_embeddings = self.get_crossattn_state_embeddings(states)
        action_embeddings = self.get_crossattn_action_embeddings(actions, attention_mask=attention_mask)
        return_embeddings = self.get_crossattn_return_embeddings(returns_to_go)
        reward_embeddings = self.get_crossattn_reward_embeddings(rewards)
        return state_embeddings, action_embeddings, return_embeddings, reward_embeddings
    
    def get_crossattn_action_embeddings(self, action, attention_mask=None):
        if self.is_discrete:
            action = action.flatten(start_dim=1)
        emb = self.crossattn_embed_action(action)
        return emb
    
    def get_crossattn_state_embeddings(self, state, mod_vectors=None):
        if self.img_is_encoded and len(state.shape) == 3: 
            return state
        if len(state.shape) > 4:
            # is_image_space
            batch_size, seq_len = state.shape[0], state.shape[1]
            state = state.reshape(-1, *self.observation_space.shape)
            # for images, we don't maintain separate encoder
            return self.embed_image(state, mod_vectors=mod_vectors).reshape(batch_size, seq_len, self.config.hidden_size)
        return self.crossattn_embed_state(state)

    def get_crossattn_return_embeddings(self, returns):
        return_embeddings = None
        if self.rtg_condition:
            if self.symlog_transform: 
                returns = torch.sign(returns) * torch.log(1 + torch.abs(returns))
            return_embeddings = self.crossattn_embed_return(returns)
        return return_embeddings
    
    def get_crossattn_reward_embeddings(self, rewards): 
        reward_embeddings = None
        if self.reward_condition:
            if self.symlog_transform: 
                rewards = torch.sign(rewards) * torch.log(1 + torch.abs(rewards))
            reward_embeddings = self.crossattn_embed_reward(rewards)
        return reward_embeddings

    def get_crossattn_time_embeddings(self, timesteps, attention_mask=None):
        return self.crossattn_embed_timestep(timesteps)

    def reinit_weights(self):
        if hasattr(self, "embed_image") and self.embed_image is not None: 
            for name, module in self.named_modules():
                if "embed_image" in name:
                    continue
                self._init_weights(module)
        else:          
            self.apply(self._init_weights)
            
    def _init_weights(self, module):
        """
        Initialize the weights. 
        From: https://github.com/huggingface/transformers/blob/a0857740c0e6127485c11476650314df3accc2b6/src/transformers/models/decision_transformer/modeling_decision_transformer.py#L445
        """
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))
      

class DiscreteCacheDTModel(CacheDTModel, DiscreteDTModel):

    def __init__(self, config, observation_space, action_space, **kwargs):
        super().__init__(config, observation_space, action_space, **kwargs)
        if self.separate_ca_embed: 
            del self.crossattn_embed_action
            self.crossattn_embed_action_disc = copy.deepcopy(self.embed_action_disc)
            self.post_init()

    def get_crossattn_action_embeddings(self, action, attention_mask=None):
        return self.crossattn_embed_action(action, attention_mask)

    def crossattn_embed_action(self, actions, attention_mask=None):
        # tokenize and embeds generated discrete tokens
        if self.tokenize_a and actions.is_floating_point() and not self.tok_a_target_only:
            # tokenize only for continuous actions (works, but suboptimal)
            actions = self.tokenize_actions(actions)
        if self.action_pad_token is not None:
            actions[attention_mask == 0] = self.action_pad_token
        act_embeds = self.crossattn_embed_action_disc(actions)
        if self.a_pos_embds:
            pos = torch.arange(act_embeds.shape[2], device=act_embeds.device)
            act_embeds = act_embeds + self.embed_act_pos(pos)
        return act_embeds
    
    def get_crossattn_return_embeddings(self, returns):
        if self.tokenize_rtg and not self.tok_rtg_target_only:
            # "discretize" returns
            returns = self.tokenize_rtgs(returns)
            # nn.Embedding preserves original shape + latent dimension. Remove excess dimension
            return super().get_crossattn_return_embeddings(returns).squeeze(2)
        return super().get_crossattn_return_embeddings(returns)
    
    def get_crossattn_reward_embeddings(self, rewards):
        if self.tokenize_r:
            # "discretize" rewards
            rewards = self.tokenize_rewards(rewards)
            return super().get_crossattn_reward_embeddings(rewards).squeeze(2)
        return super().get_crossattn_reward_embeddings(rewards)

    def get_crossattn_state_embeddings(self, state, mod_vectors=None):
        if self.img_is_encoded and len(state.shape) == 3: 
            return state
        if len(state.shape) > 4:
            # is_image_space --> [B x T x C x W x H] 
            batch_size, seq_len, obs_shape = state.shape[0], state.shape[1], state.shape[2:]
            state = state.reshape(-1, *obs_shape)
            if self.patch_size is not None: 
                # patchify -->  [BT x D x P x P]
                image_emb = self.embed_image(state).permute(0, 2, 3, 1)
                if self.num_learned_s_tok is not None: 
                    # employ token learner
                    image_emb = image_emb.reshape(batch_size * seq_len, -1, self.config.hidden_size)
                    image_emb = self.s_token_learner(image_emb)
                # reshape to [B X T x P * P x D] 
                image_emb = image_emb.reshape(batch_size, seq_len, -1, self.config.hidden_size)
                image_emb = image_emb + self.embed_patch_pos
                return image_emb
                        
            return self.embed_image(state, mod_vectors=mod_vectors).reshape(batch_size, seq_len, self.config.hidden_size)
        
        if self.tokenize_s:
            # "discretize" states
            state = self.tokenize_states(state)
            # nn.Embedding preserves original shape + latent dimension. Remove excess dimension
            return super().get_crossattn_state_embeddings(state).squeeze(2)
        return self.crossattn_embed_state(state)

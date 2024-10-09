import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ortho_group
from transformers import AutoModel
from .online_decision_transformer_model import OnlineDecisionTransformerModel
from .discrete_decision_transformer_model import DiscreteDTModel
from .model_utils import make_sinusoidal_embd
from ...tokenizers_custom import make_tokenizer


class FrozenHopfield(nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_dim,
        embeddings,
        beta, 
        trainable=False, 
        orthog=False,
        sinusoidal=False, 
        center=False, 
        std_kind="ratio", 
        top_k=None,
        lookup=False
    ):
        """
        Adjusted from: https://github.com/ml-jku/helm/blob/main/model.py

        """
        super(FrozenHopfield, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.beta = beta
        self.orthog = orthog
        self.std_kind = std_kind
        self.sinusoidal = sinusoidal
        self.top_k = top_k
        self.center = center
        self.lookup = lookup 
        self.register_buffer("embeddings", embeddings)
        # store normed embeddings, otherwise always doing recalculation
        if self.center: 
            self.register_buffer("embed_mean", embeddings.mean(dim=0))
            self.register_buffer("embeddings_norm", (embeddings - self.embed_mean).norm(dim=-1))
        else: 
            self.register_buffer("embeddings_norm", embeddings.norm(dim=-1))
        if sinusoidal:
            # input_dim defines n_positions
            self.proj = make_sinusoidal_embd(input_dim, embedding_dim)
            for p in self.proj.parameters():
                p.requires_grad = False
        else: 
            if self.std_kind == "ratio": 
                std = np.sqrt(input_dim / embedding_dim) 
            elif self.std_kind == "input": 
                std = 1 / np.sqrt(input_dim) 
            else: 
                std = 1 / np.sqrt(embedding_dim)
            rand_matrix = torch.normal(mean=0.0, std=std, size=(embedding_dim, input_dim))
            if orthog: 
                rand_matrix = torch.Tensor(ortho_group.rvs(embedding_dim))[:, :input_dim]
            self.proj = torch.nn.Parameter(rand_matrix, requires_grad=trainable)

    def forward(self, x):
        # torch.matmul handles broadcasting --> works with 2D (bs x input_dim) or 3D (bs x seq_len x input_dim)
        if self.sinusoidal: 
            x = self.proj(x.long()).flatten(2)
        elif self.lookup: 
            x = self.proj.T[x].flatten(2)
        else: 
            x = torch.matmul(x, self.proj.T)
        embeds_maybe_centered = self.embeddings - self.embed_mean if self.center else self.embeddings            
        similarities = torch.matmul(x, embeds_maybe_centered.T) / (
            x.norm(dim=-1, keepdim=True) * self.embeddings_norm.unsqueeze(0) + 1e-8
        )
        if self.top_k is not None:
            # only consider top-k in sim computation
            topk_values, topk_idx = torch.topk(similarities, self.top_k, dim=-1)
            if self.top_k > 100: 
                # memory efficient, slower version
                mask = torch.zeros_like(similarities, dtype=torch.bool).scatter_(-1, topk_idx, 1)
                masked_similarities = torch.where(mask, similarities, float('-inf'))
                softm = torch.softmax(self.beta * masked_similarities, dim=-1)
                out = torch.matmul(softm, self.embeddings)
            else: 
                # lots of redudndancy in embedding vectors here
                softm = torch.softmax(self.beta * topk_values, dim=-1)
                topk_embeds = self.embeddings[topk_idx]
                out = (topk_embeds * softm.unsqueeze(-1)).sum(dim=-2)
        else: 
            softm = torch.softmax(self.beta * similarities, dim=-1)
            out = torch.matmul(softm, self.embeddings)
        return out
    

class BaseHelmDTModel(OnlineDecisionTransformerModel):

    def __init__(self, config, observation_space, action_space, beta=1,
                 on_actions=True, on_states=True, on_rtgs=True, on_rewards=True,
                 orthog=False, sinusoidal=False, std_kind="ratio", 
                 pretrained_lm="bert-base-uncased", top_k=None, **kwargs):
        """
        Initialize the HELMDecisionTransformerModel.
        Uses random projections to map states/actions/rewards/rtgs into embedding space of pre-trained LM. 
        Inputs dimes of 1 (actions, rewards, rtg) do not make sense for random projection, as it just scales vectors. 
        Therefore, discretise (if required) and convert to one-hot representation to retrieve well-defined LM tokens.    
    
        Args:
            config (dict): Huggingface config.
            observation_space (gym.Space): The observation space of the environment.
            action_space (gym.Space): The action space of the environment.
            beta (float, optional): The beta value for FrozenHopfield embeddings. Defaults to 1.
            on_actions (bool, optional): Whether to use FrozenHopfield embeddings for actions. Defaults to True.
            on_states (bool, optional): Whether to use FrozenHopfield embeddings for states. Defaults to True.
            on_rtgs (bool, optional): Whether to use FrozenHopfield embeddings for returns. Defaults to True.
            on_rewards (bool, optional): Whether to use FrozenHopfield embeddings for rewards. Defaults to True.
            pretrained_lm (str, optional): The name of the pretrained language model. Defaults to "bert-base-uncased".
            **kwargs: Additional keyword arguments.
        """
        config = self._prepare_config(config, pretrained_lm)
        super().__init__(config, observation_space, action_space, **kwargs)
        self.on_actions = on_actions
        self.on_states = on_states
        self.on_rtgs = on_rtgs
        self.on_rewards = on_rewards
        self.orthog = orthog
        self.std_kind = std_kind
        self.sinusoidal = sinusoidal
        self.top_k = top_k
        self.embed_ln = nn.Identity()
        del self.encoder
        self.encoder = AutoModel.from_pretrained(pretrained_lm, config=config, ignore_mismatched_sizes=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        word_embeds = self.encoder.wte if hasattr(self.encoder, "wte") else self.encoder.embeddings.word_embeddings
        n_tokens = word_embeds.num_embeddings
        embedding_dim = word_embeds.embedding_dim
        embeddings = word_embeds(torch.arange(n_tokens))
        
        if self.on_states:
            self._setup_state_projection(embedding_dim, self.config.state_dim, embeddings,
                                         beta=beta, orthog=orthog, std_kind=std_kind, sinusoidal=sinusoidal, 
                                        top_k=top_k)
        if self.on_actions:
            # if act_dim == 1 --> requires discretization --> otherwise random projection will always be the same
            self._setup_action_projection(embedding_dim, self.config.act_dim, embeddings,
                                          beta=beta, orthog=orthog, std_kind=std_kind, top_k=top_k)
        if self.on_rtgs:
            self._setup_rtg_projection(embedding_dim, embeddings, 
                                       beta=beta, orthog=orthog, std_kind=std_kind,
                                       sinusoidal=sinusoidal, top_k=top_k)
        if self.on_rewards:
            self._setup_reward_projection(embedding_dim, embeddings, 
                                          beta=beta, orthog=orthog, std_kind=std_kind,
                                          sinusoidal=sinusoidal, top_k=top_k)
    
    def _setup_state_projection(self, embedding_dim, state_dim, embeddings, beta, orthog, std_kind,
                                sinusoidal, top_k): 
        del self.embed_state
        self.embed_state = FrozenHopfield(embedding_dim, state_dim, embeddings, beta=beta, 
                                          orthog=orthog, std_kind=std_kind, sinusoidal=sinusoidal,
                                          top_k=top_k)
    
    def _setup_action_projection(self, embedding_dim, act_dim, embeddings, beta,
                                 orthog, std_kind, top_k): 
        del self.embed_action
        self.embed_action = FrozenHopfield(embedding_dim, act_dim, embeddings, beta=beta,
                                           orthog=orthog, std_kind=std_kind, top_k=top_k)
    
    def _setup_reward_projection(self, embedding_dim, embeddings, beta, orthog, std_kind,
                                 sinusoidal, top_k): 
        del self.embed_rewards
        self.embed_rewards = FrozenHopfield(embedding_dim, 1, embeddings, beta=beta,
                                            orthog=orthog, std_kind=std_kind, sinusoidal=sinusoidal,
                                            top_k=top_k)
        
    def _setup_rtg_projection(self, embedding_dim, embeddings, beta, orthog, std_kind, sinusoidal, top_k):
        del self.embed_return
        self.embed_return = FrozenHopfield(embedding_dim, 1, embeddings, beta=beta,
                                           orthog=orthog, std_kind=std_kind, sinusoidal=sinusoidal, 
                                           top_k=top_k)
    
    def _prepare_config(self, config, pretrained_lm):
        # prepare config for compatibility reasons 
        if "distilbert" in pretrained_lm:
            config.n_positions = config.max_position_embeddings
            config.n_layer = config.num_hidden_layers
            config.n_head = config.num_attention_heads
            config.activation_function = config.activation
            config.resid_pdrop = config.dropout
            config.embd_pdrop = config.dropout
            config.attn_pdrop = config.attention_dropout
            config.layer_norm_epsilon = 1e-05
            config.n_embd = config.vocab_size
            config.n_inner = None
            config.scale_attn_weights = False
            config.scale_attn_by_inverse_layer_idx = False
            config.reorder_and_upcast_attn = False
            config.use_cache = False
            config.hidden_size = config.dim
        elif "bert" in pretrained_lm:
            # add kwargs for BERT 
            config.n_positions = config.max_position_embeddings
            config.n_layer = config.num_hidden_layers
            config.n_head = config.num_attention_heads
            config.activation_function = config.hidden_act
            config.resid_pdrop = config.hidden_dropout_prob
            config.embd_pdrop = config.hidden_dropout_prob
            config.attn_pdrop = config.attention_probs_dropout_prob
            config.layer_norm_epsilon = config.layer_norm_eps
            config.n_embd = config.vocab_size
            config.n_inner = None
            config.scale_attn_weights = False
            config.scale_attn_by_inverse_layer_idx = False
            config.reorder_and_upcast_attn = False
        return config
    

class HelmDTModel(BaseHelmDTModel):
    
    def __init__(self, config, observation_space, action_space, rtg_tok_kwargs=None, r_tok_kwargs=None, **kwargs):
        # Reward / RTG always need to be tokenized
        # tokenization kwargs need to be defined before calling super().__init__                
        self.rtg_tok_kwargs = rtg_tok_kwargs or {}
        self.r_tok_kwargs = r_tok_kwargs or {}
        super().__init__(config, observation_space, action_space, **kwargs)
        
    def _setup_reward_projection(self, embedding_dim, embeddings, beta, orthog, std_kind,
                                 sinusoidal, top_k): 
        del self.embed_rewards
        del self.predict_reward
        r_tok_kind = self.r_tok_kwargs.pop('kind', 'minmax')
        num_r_bins = self.r_tok_kwargs.get('vocab_size', 100)
        self.r_tok_kwargs["vocab_size"] = num_r_bins
        self.r_tokenizer = make_tokenizer(r_tok_kind, self.r_tok_kwargs)
        self.embed_rewards = FrozenHopfield(embedding_dim, num_r_bins, embeddings, beta=beta, 
                                            orthog=orthog, std_kind=std_kind, sinusoidal=sinusoidal, 
                                            top_k=top_k, lookup=True)
        self.predict_reward = nn.Linear(self.config.hidden_size, num_r_bins)

        
    def _setup_rtg_projection(self, embedding_dim, embeddings, beta, orthog, std_kind, sinusoidal, top_k):
        del self.embed_return
        del self.predict_return
        rtg_tok_kind = self.rtg_tok_kwargs.pop('kind', 'minmax')
        num_rtg_bins = self.rtg_tok_kwargs.get('vocab_size', 100)
        self.rtg_tok_kwargs["vocab_size"] = num_rtg_bins
        self.rtg_tokenizer = make_tokenizer(rtg_tok_kind, self.rtg_tok_kwargs)
        self.embed_return = FrozenHopfield(embedding_dim, num_rtg_bins, embeddings, beta=beta,
                                           orthog=orthog, std_kind=std_kind, sinusoidal=sinusoidal, 
                                           top_k=top_k, lookup=True)
        self.predict_return = nn.Linear(self.config.hidden_size, num_rtg_bins) 
    
    @torch._dynamo.disable
    def get_return_embeddings(self, returns):
        return_embeddings = None
        if self.rtg_condition:
            if self.symlog_transform: 
                returns = torch.sign(returns) * torch.log(1 + torch.abs(returns))
            if self.on_rewards and not self.sinusoidal: 
                returns = self.rtg_tokenizer.tokenize(returns)
                # returns = torch.nn.functional.one_hot(returns, num_classes=self.rtg_tokenizer.vocab_size).float()
            return_embeddings = self.embed_return(returns).squeeze(2)
        return return_embeddings
    
    @torch._dynamo.disable
    def get_reward_embeddings(self, rewards): 
        reward_embeddings = None
        if self.reward_condition:
            if self.symlog_transform: 
                rewards = torch.sign(rewards) * torch.log(1 + torch.abs(rewards))
            if self.on_rewards and not self.sinusoidal: 
                rewards = self.r_tokenizer.tokenize(rewards)
                # rewards = torch.nn.functional.one_hot(rewards, num_classes=self.r_tokenizer.vocab_size).float()
            reward_embeddings = self.embed_rewards(rewards).squeeze(2)
        return reward_embeddings


class DiscreteHelmDTModel(BaseHelmDTModel, DiscreteDTModel):
    
    def _setup_action_projection(self, embedding_dim, act_dim, embeddings, beta, orthog, std_kind, top_k):
        del self.embed_action_disc
        # needs to be action channels --> expects a one-hot vector input
        # actions are already "discretized"
        self.embed_action_disc = FrozenHopfield(embedding_dim, self.action_channels + 1, embeddings, 
                                                beta=beta, orthog=orthog, std_kind=std_kind, top_k=top_k)
        
    def _setup_state_projection(self, embedding_dim, state_dim, embeddings, beta, orthog, std_kind, 
                                sinusoidal, top_k): 
        del self.embed_state
        if self.tokenize_s and self.s_tokenizer.one_hot:
            sinusoidal = False
        if self.tokenize_s: 
            if self.s_tokenizer.one_hot: 
                state_dim = self.s_tokenizer.vocab_size * state_dim
            elif sinusoidal:
                # distribute
                embedding_dim //= state_dim
                state_dim = self.s_tokenizer.vocab_size
        self.embed_state = FrozenHopfield(embedding_dim, state_dim, embeddings, beta=beta,
                                          orthog=orthog, std_kind=std_kind, sinusoidal=sinusoidal, 
                                          top_k=top_k)
        
    def _setup_reward_projection(self, embedding_dim, embeddings, beta, orthog, std_kind, sinusoidal, top_k): 
        del self.embed_rewards
        r_dim = self.r_tokenizer.vocab_size if self.tokenize_r else 1
        self.embed_rewards = FrozenHopfield(embedding_dim, r_dim, embeddings, beta=beta, 
                                            orthog=orthog, std_kind=std_kind, sinusoidal=sinusoidal, 
                                            top_k=top_k, lookup=True)
        
    def _setup_rtg_projection(self, embedding_dim, embeddings, beta, orthog, std_kind, sinusoidal, top_k):
        del self.embed_return
        rtg_dim = self.rtg_tokenizer.vocab_size if self.tokenize_rtg else 1
        self.embed_return = FrozenHopfield(embedding_dim, rtg_dim, embeddings, beta=beta,
                                           orthog=orthog, std_kind=std_kind, sinusoidal=sinusoidal, 
                                           top_k=top_k, lookup=True)        
        
    @torch._dynamo.disable
    def embed_action(self, actions, attention_mask=None):
        # tokenize and embeds generated discrete tokens
        if self.tokenize_a and actions.is_floating_point():
            # tokenize only for continuous actions (works, but suboptimal)
            actions = self.tokenize_actions(actions)
        if self.action_pad_token is not None:
            actions[attention_mask == 0] = self.action_pad_token
            
        if self.on_actions: 
            # convert actions to one-hot
            actions = torch.nn.functional.one_hot(actions, num_classes=self.action_channels + 1).float()
            
        act_embeds = self.embed_action_disc(actions)
        if self.a_pos_embds:
            pos = torch.arange(act_embeds.shape[2], device=act_embeds.device)
            act_embeds = act_embeds + self.embed_act_pos(pos)
        return act_embeds

    @torch._dynamo.disable
    def tokenize_states(self, states):
        return super().tokenize_states(states)

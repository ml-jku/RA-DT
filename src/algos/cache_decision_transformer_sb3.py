import math
import copy
import collections
import torch
import torchmetrics
import numpy as np
from tqdm import tqdm
from pathlib import Path
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.save_util import load_from_zip_file
from .universal_decision_transformer_sb3 import UDT
from .discrete_decision_transformer_sb3 import DiscreteDecisionTransformerSb3
from .models.model_utils import sample_from_logits
from .agent_utils import aggregate_embeds, get_param_count, add_gaussian_noise
from ..buffers.trajectory import Trajectory
from ..buffers.cache import Cache
from ..buffers.buffer_utils import discount_cumsum_torch, compute_start_end_context_idx
from ..utils.misc import make_retrieved_states_plot
from ..envs.builder import extract_full_env_names


class CacheDecisionTransformerSb3(UDT):

    def __init__(
        self, 
        policy,
        env,
        ret_fraction=0.2,
        top_k_ret=1,
        cache_len=1,
        cache_steps=1,
        p_mask=0,
        p_context_mask=0,
        query_dropout=0,
        eval_ret_steps=1,
        action_kind="mix",
        representation_type="last",
        agg_token="s",
        learnable_ret=False,
        rel_ret_fraction=False,
        reinit_policy=False, 
        sep_eval_cache=False,
        shared_eval_cache=False,
        record_ret_stats=True,
        full_context_len=False,
        dynamic_context_len=False,
        rand_first_chunk=False,
        ret_mask_rtg=False,
        ret_rtg_condition=True,
        ret_time_embds=None,
        plot_ret_freq=None, 
        cache_data_paths=None,
        cache_kwargs=None,
        eval_cache_kwargs=None,
        max_embed_len=None,
        future_cache_len=None,
        chunk_len=None,
        precalc_save_dir=None,
        precalc_load_dir=None,
        retriever=None,
        demo_tasks=None,
        demo_data_paths=None,
        ret_load_path=None,
        p_context_drop=None,
        **kwargs
    ):
        self.cache_kwargs = cache_kwargs if cache_kwargs is not None else {}
        self.eval_cache_kwargs = eval_cache_kwargs if eval_cache_kwargs is not None else {}
        self.cache_data_paths = cache_data_paths
        self.learnable_ret = learnable_ret
        self.cache_len = cache_len 
        self.future_cache_len = future_cache_len if future_cache_len is not None else cache_len
        self.representation_type = representation_type
        self.record_ret_stats = record_ret_stats
        self.cache_steps = cache_steps
        self.full_context_len = full_context_len
        self.dynamic_context_len = dynamic_context_len
        self.rand_first_chunk = rand_first_chunk
        self.agg_token = agg_token
        self.max_embed_len = max_embed_len
        self.query_dropout = query_dropout
        self.ret_mask_rtg = ret_mask_rtg
        self.ret_rtg_condition = ret_rtg_condition
        self.ret_time_embds = ret_time_embds
        self.retriever = retriever
        self.shared_eval_cache = shared_eval_cache
        self.sep_eval_cache = sep_eval_cache
        self.demo_data_paths = demo_data_paths
        self.ret_load_path = ret_load_path        
        self.demo_tasks = extract_full_env_names(demo_tasks) if demo_tasks is not None else None
        # force global trj ids
        kwargs["replay_buffer_kwargs"] = {**kwargs.get("replay_buffer_kwargs", {}), "use_global_trj_ids": True}
        super().__init__(policy, env, **kwargs)
        self.ret_fraction = ret_fraction
        self.rel_ret_fraction = rel_ret_fraction
        self.top_k_ret = top_k_ret
        self.action_kind = action_kind
        self.plot_ret_freq = plot_ret_freq
        self.p_context_mask = p_context_mask
        self.p_mask = p_mask
        self.chunk_len = chunk_len
        self.p_context_drop = p_context_drop
        self.eval_ret_steps = eval_ret_steps
        self.precalc_save_dir = precalc_save_dir
        self.precalc_load_dir = precalc_load_dir
        self.precalc_dict = None
        self.eval_task = None
        if self.precalc_save_dir is not None:
            self.precalc_save_dir = Path(self.precalc_save_dir)
            self.precalc_save_dir.mkdir(parents=True, exist_ok=True)
            self.precalc_dict = collections.defaultdict(list)
        elif self.precalc_load_dir is not None:
            precalc_dict = np.load(f"{self.precalc_load_dir}/seed_{self.seed}.npz")
            self.precalc_dict = {key: precalc_dict[key] for key in precalc_dict.files}
            self.precalc_counter = 0 
        self.eval_ret_steps_counter = 0
        self.eval_ret_context_trjs = None
        self.pred_counter = 0
        if reinit_policy: 
            print("Re-initing policy weights.")
            self.policy.reinit_weights()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._setup_retriever()
        self._setup_cache()   
        self._eval_cache = None
        if self.shared_eval_cache: 
            self._setup_eval_cache(None, None, None)
        if self.demo_tasks is not None: 
            self._setup_demo_cache()
            
    def _setup_retriever(self): 
        self.is_pretrained_retriever = False
        if self.learnable_ret: 
            maybe_compiled = False
            if self.retriever is None: 
                self.retriever = copy.deepcopy(self.policy)
                self.is_pretrained_retriever = True
                # already compiled model
                maybe_compiled = True
                if self.ret_load_path is not None:
                    self.load_retriever_weights(self.ret_load_path)
            else: 
                self.retriever = self.retriever.to(self.device)
            for param in self.retriever.parameters():
                param.requires_grad = False
            self.retriever.eval()
            print(self.retriever)
            if torch.__version__ >= "2.0.0" and self.compile and not maybe_compiled: 
                self.retriever = torch.compile(self.retriever)
            if self.ret_mask_rtg: 
                self.retriever.mask_rtg = True
            if not self.ret_rtg_condition: 
                self.retriever.rtg_condition = False
            if self.ret_time_embds is not None: 
                self.retriever.use_time_embds = self.ret_time_embds

    def _setup_cache(self):
        self.cache_kwargs.update({
            "device": self.device, 
            "n_envs": self.n_envs,
            "target_return": self.buffer_target_return,
            "max_len_type": self.buffer_max_len_type,
            "learnable_ret": self.learnable_ret,
            "cache_context_len": self.cache_len,
            "future_context_len": self.future_cache_len,
            "cache_steps": self.cache_steps, 
            "full_context_len": self.full_context_len,
            "dynamic_context_len": self.dynamic_context_len,
            "rand_first_chunk": self.rand_first_chunk,
            "use_global_trj_ids": True, 
            "max_act_dim": self.replay_buffer.max_act_dim,
            "max_state_dim": self.replay_buffer.max_state_dim,
        })
        if hasattr(self.policy_class, "config"):
            self.cache_kwargs["context_len"] = self.policy_class.config.max_length
            self.cache_kwargs["max_len"] = self.policy_class.config.max_ep_len
        share_trjs = self.cache_kwargs.pop("share_trjs", False)
        self.cache = Cache(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            **self.cache_kwargs,
        )
        if share_trjs:
            # share trajectories to save memory and dataloading time
            assert self.replay_buffer is not None, "Need to pass replay_buffer to share trjs."
            self.cache.init_from_existing_buffer(self.replay_buffer)
        else:
            self.cache.init_buffer_from_dataset(self.cache_data_paths)
        if len(self.cache) > 0:  
            keys, values = self.construct_cache_keys_and_values(self.cache)
            self.cache.setup_cache(keys, values)

    def _setup_eval_cache(self, keys, values, total_return, demo_cache=False):
        self.eval_cache_kwargs.update({
            "device": self.device, "n_envs": self.n_envs,
            "target_return": self.buffer_target_return,
            "max_len_type": self.buffer_max_len_type,
            "cache_steps": self.cache_steps,
        })
        if hasattr(self.policy_class, "config"):
            self.eval_cache_kwargs["context_len"] = self.policy_class.config.max_length
            self.eval_cache_kwargs["max_len"] = self.policy_class.config.max_ep_len
        if self.sep_eval_cache: 
            self._eval_cache = Cache(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                **self.eval_cache_kwargs
            )
            if not self._eval_cache.should_store_trj(total_return, self.reward_scale): 
                return 
            self._eval_cache.setup_cache(keys, values)
        else: 
            self._eval_cache = copy.copy(self.cache)
            self._eval_cache.update_attributes(self.eval_cache_kwargs)
                
    def _setup_demo_cache(self):
        assert self.demo_tasks is not None and self.demo_data_paths is not None
        self.eval_cache_kwargs.update({
            "device": self.device, 
            "n_envs": self.n_envs,
            "target_return": self.buffer_target_return,
            "max_len_type": self.buffer_max_len_type,
            "learnable_ret": self.learnable_ret,
            "cache_context_len": self.cache_len,
            "future_context_len": self.future_cache_len,
            "cache_steps": self.cache_steps, 
            "full_context_len": self.full_context_len,
            "dynamic_context_len": self.dynamic_context_len,
            "rand_first_chunk": self.rand_first_chunk,
            "use_global_trj_ids": True, 
            "max_act_dim": self.replay_buffer.max_act_dim,
            "max_state_dim": self.replay_buffer.max_state_dim,
            "num_workers": 0,
            "prefetch_factor": None,
        })
        if hasattr(self.policy_class, "config"):
            self.eval_cache_kwargs["context_len"] = self.policy_class.config.max_length
            self.eval_cache_kwargs["max_len"] = self.policy_class.config.max_ep_len
        self._demo_cache = Cache(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            **self.eval_cache_kwargs,
        )
        self._demo_cache.init_buffer_from_dataset(self.demo_data_paths)
        keys, values = self.construct_cache_keys_and_values(self._demo_cache)
        self._demo_cache.setup_cache(keys, values)
        
    @property
    def eval_cache(self):
        if self.demo_tasks is not None and self.eval_task in self.demo_tasks:
            return self._demo_cache  
        return self._eval_cache
    
    def sample_batch_with_context(self, cache, batch_size):
        replay_data = cache.sample(
            batch_size=batch_size,
            weight_by=self.buffer_weight_by,
            env=self._vec_normalize_env,
            top_k=self.buffer_topk
        )
        if self.state_mean is not None and self.state_std is not None:
            replay_data.observations = (replay_data.observations - self.state_mean) / self.state_std
            replay_data.context_observations = (replay_data.context_observations - self.state_mean) / self.state_std
        if self.reward_scale != 1:
            replay_data.rewards_to_go = replay_data.rewards_to_go / self.reward_scale
            replay_data.rewards = replay_data.rewards / self.reward_scale
            replay_data.context_rewards_to_go = replay_data.context_rewards_to_go / self.reward_scale
            replay_data.context_rewards = replay_data.context_rewards / self.reward_scale
            replay_data.total_returns = replay_data.total_returns / self.reward_scale
    
        return replay_data
    
    @torch.no_grad()
    def construct_cache_keys_and_values(self, cache): 
        # iterate sequences, compute hidden states and store them with actions in the cache
        self.policy.eval()
        num_samples = sum(
            [len(t) // self.cache_steps if isinstance(t, Trajectory) else cache.trajectory_lengths[str(t)] // self.cache_steps
             for t in cache.trajectories]
        )
        # to speed-up computation
        num_batches = math.ceil(num_samples / self.batch_size)            
        embed_model = self.retriever if self.retriever is not None else self.policy
        keys, values = [], collections.defaultdict(list)
        for _ in tqdm(range(num_batches), desc="Caching top trajectories"):
            batch = self.sample_batch_with_context(cache, self.batch_size)
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                policy_output = embed_model(
                    states=batch.observations,
                    actions=batch.actions,
                    rewards=batch.rewards,
                    returns_to_go=batch.rewards_to_go,
                    timesteps=batch.timesteps.long(),
                    attention_mask=batch.attention_mask,
                    return_dict=True,
                    with_log_probs=self.stochastic_policy,
                    deterministic=False,
                    ddp_kwargs=self.ddp_kwargs,
                )
            # extract last hidden state from policy_output - one position before last action token
            key = self.extract_representation(policy_output, embed_model, 
                                              states=batch.observations, attention_mask=batch.attention_mask)
            # append hidden state and corresponding action
            keys.append(key.detach().cpu())
            if self.learnable_ret:
                # store "past" + "future" context
                values["actions"].append(batch.context_actions.detach().cpu())
                values["states"].append(batch.context_observations.detach().cpu())
                values["returns_to_go"].append(batch.context_rewards_to_go.detach().cpu())
                values["rewards"].append(batch.context_rewards.detach().cpu())
                values["timesteps"].append(batch.context_timesteps.detach().cpu())
                values["attention_mask"].append(batch.context_attention_mask.detach().cpu())
                values["task_ids"].append(batch.task_ids.detach().cpu()) 
                values["total_return"].append(batch.total_returns.detach().cpu()) 
                values["trj_ids"].append(batch.trj_ids.detach().cpu())            
                values["trj_seeds"].append(batch.trj_seeds.detach().cpu())
            else: 
                values["actions"].append(batch.actions[:, -self.cache_len].detach().cpu())
                values["states"].append(batch.observations[:, -self.cache_len].detach().cpu())
        # remove to free ram
        self.cache._trajectories = []
        self.cache._trajectory_lengths = {}
        del self.cache.trj_iterator, self.cache.trj_loader, self.cache.trj_dataset
        
        # make cache
        print("Converting keys and values to numpy.")
        keys = torch.concat(keys, dim=0).numpy()
        for k, v in values.items(): 
            values[k] = torch.concat(v, dim=0)
        if len(keys.shape) > 2:
            keys = keys.squeeze(1)
        self.policy.train()
        return keys, values
    
    def extract_representation(self, output, model, query_dropout=0, 
                               states=None, attention_mask=None, chunk_len=None):
        if chunk_len is not None:
            assert self.representation_type == "mean", "Chunk len only makes sense with aggregation."
        if self.representation_type == "last": 
            return self.extract_last_hidden_state(output, model)
        elif self.representation_type == "mean": 
            return aggregate_embeds(
                output.last_hidden_state, 
                model.tok_to_pos, 
                attention_mask=attention_mask,
                max_embed_len=self.max_embed_len if self.max_embed_len is not None else self.cache_steps,
                agg_token=self.agg_token, 
                dropout=query_dropout,
                chunk_len=chunk_len
            )
        elif self.representation_type == "embed":
            return self.extract_from_layer(output, model, layer=0)
        elif self.representation_type == "first": 
            return self.extract_from_layer(output, model, layer=1)
        elif self.representation_type == "raw_state": 
            return states[:, -1]
        raise NotImplementedError(f"'{self.representation_type}' not yet implemented.")
    
    def extract_last_hidden_state(self, output, model):
        # extract last hidden state prior to the last action token
        last_hidden_state = output.last_hidden_state.reshape(
            output.last_hidden_state.shape[0],
            model.config.max_length,
            len(model.tok_to_pred_pos),
            model.hidden_size
        ).permute(0, 2, 1, 3)
        return last_hidden_state[:, model.tok_to_pred_pos['a'], -1]

    def extract_from_layer(self, output, model, layer=0, tok=None):
        # extract latent representation of state prior to the last action token in particular layer
        # embedding layer == 0
        latent_rep = output.hidden_states[layer]
        latent_rep = latent_rep.reshape(
            latent_rep.shape[0],
            model.config.max_length,
            len(model.tok_to_pred_pos),
            model.hidden_size
        ).permute(0, 2, 1, 3)
        pos = model.tok_to_pred_pos["a"] if tok is None else model.tok_to_pos[tok]
        return latent_rep[:, pos, -1]
    
    def _train_step(self, batch_size):
        """
        Performs a single train step. 
        Updates the orignal function to include the retrieval of context trajectories and passes them 
        to the policy.
        
        Assumes that this function is only called, in case self.learnable_ret == True.

        Args:
            batch_size: Int. Batch size to use for training.

        Returns:
            metrics: Dict.
        """        
        metrics = {}
        observations, actions, next_observations, rewards, rewards_to_go, timesteps, attention_mask, dones, task_ids,\
            trj_ids, action_targets, action_mask, prompt, total_returns, trj_seeds = self.sample_batch(batch_size)
        
        # retrieve relevant context
        context_trjs = None
        if self.learnable_ret and not (self.p_context_drop is not None and np.random.rand() < self.p_context_drop): 
            distances_ret, idx_ret, context_trjs = self.retrieve_context_trjs(
                self.cache, observations, actions, rewards, rewards_to_go,
                timesteps, attention_mask, prompt, task_ids, trj_ids, 
                p_mask=self.p_context_mask, query_dropout=self.query_dropout,
                precalc_dict=self.precalc_dict if self.precalc_load_dir is not None else None,
                total_returns=total_returns, trj_seeds=trj_seeds
            )
            if self.precalc_save_dir is not None:
                self.precalc_dict["idx"].append(idx_ret)
                self.precalc_dict["distances"].append(distances_ret)
                # early return, no updates during precalculation
                return metrics
            
        if self.record_ret_stats and self.num_timesteps % 1000 == 0 and context_trjs is not None: 
            ret_metrics = self._record_retrieval_stats(context_trjs, task_ids, trj_ids, trj_seeds,
                                                       total_returns, observations)
            metrics.update(ret_metrics)
        if self.p_mask > 0: 
            # option for masking of input_trjs - doing it directly in buffer affects retrieval
            mask = torch.bernoulli(torch.full(attention_mask.shape, self.p_mask, device=attention_mask.device))
            attention_mask = attention_mask * (1 - mask)

        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            policy_output = self.policy(
                states=observations,
                actions=actions,
                rewards=rewards,
                returns_to_go=rewards_to_go,
                timesteps=timesteps.long(),
                attention_mask=attention_mask,
                return_dict=True,
                with_log_probs=self.stochastic_policy,
                deterministic=False,
                prompt=prompt,
                task_id=self.current_task_id_tensor,
                ddp_kwargs=self.ddp_kwargs,
                context_trjs=context_trjs
            )
        action_log_probs, action_log_probs_masked, entropy_masked = None, None, None
        if self.stochastic_policy:
            action_log_probs = policy_output.action_log_probs
            action_log_probs_masked = action_log_probs.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            metrics["action_log_probs_mean"] = action_log_probs_masked.mean().item()
            if policy_output.entropy is not None:
                entropy_masked = policy_output.entropy.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            if self.last_seq_only:
                # action_log_probs_masked is already masked. Only use last sequence for backward pass.
                is_last_seq = torch.zeros_like(attention_mask)
                is_last_seq[:, -1] = 1
                is_last_seq = is_last_seq.reshape(-1)[attention_mask.reshape(-1) > 0] > 0
                action_log_probs_masked = action_log_probs_masked[is_last_seq]
                entropy_masked = entropy_masked[is_last_seq] if entropy_masked is not None else None

        # update ent_coef
        if self.stochastic_policy and self._n_updates > self.ent_tuning_start:
            ent_coef, ent_coef_dict = self.update_entropy_coef(action_log_probs_masked, entropy=entropy_masked)
            for k, v in ent_coef_dict.items():
                metrics[k] = v
            ent_tuning = True
        else:
            ent_coef = 0
            ent_tuning = False
            
        # compute loss + update
        loss_dict = self.update_policy(
            policy_output, action_targets, attention_mask, ent_coef,
            return_targets=rewards_to_go, ent_tuning=ent_tuning,
            reward_targets=rewards, state_targets=observations, timesteps=timesteps, 
            dones=dones.float(), next_states=next_observations, action_mask=action_mask
        )
        for k, v in loss_dict.items():
            metrics[k] = v

        if (self._n_updates + 1) % 10000 == 0 and self.log_attn_maps:
            self._record_attention_maps(policy_output.attentions, step=self.num_timesteps, prefix="train")
            if policy_output.cross_attentions is not None:
                self._record_attention_maps(policy_output.cross_attentions, step=self.num_timesteps,
                                            prefix="train_cross", lower_triu=False)
        metrics["target_returns"] = rewards_to_go.mean().item()
        self._n_updates += 1
        return metrics
    
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        eval_env=None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DecisionTransformer",
        eval_log_path=None,
        reset_num_timesteps: bool = True,
    ):
        res = super().learn(total_timesteps, callback, log_interval, eval_env, eval_freq, 
                            n_eval_episodes, tb_log_name, eval_log_path, reset_num_timesteps)
        if self.precalc_save_dir is not None:
            self.precalc_dict = {k: np.concatenate(v) for k, v in self.precalc_dict.items()}         
            np.savez_compressed(self.precalc_save_dir / f"seed_{self.seed}.npz",
                                **self.precalc_dict)
        return res

    @torch.no_grad()
    def retrieve_context_trjs(self, index, states, actions, rewards, returns_to_go,
                              timesteps, attention_mask, prompt, task_ids, trj_ids=None, 
                              p_mask=0, query_dropout=0, precalc_dict=None,
                              total_returns=None, trj_seeds=None, target_return=None):
        # construct query
        if index.sample_kind is not None or precalc_dict is not None:
            # no retrieval, no need to construct query
            cache_query = states
        else: 
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                retriever_output = self.retriever(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps.long(),
                    attention_mask=attention_mask,
                    return_dict=True,
                    with_log_probs=self.stochastic_policy,
                    deterministic=False,
                    prompt=prompt,
                    task_id=self.current_task_id_tensor,
                    ddp_kwargs=self.ddp_kwargs,
                )
            # i.e., will have multiple cache_queries essentially, then probably need to reshape again. 
            cache_query = self.extract_representation(
                retriever_output, self.retriever, states=states, attention_mask=attention_mask, 
                query_dropout=query_dropout, chunk_len=self.chunk_len
            ).detach().cpu().numpy()
            if len(cache_query.shape) > 2:
                cache_query = cache_query.squeeze(1)

        if self.chunk_len is not None:
            n_chunks = states.shape[1] // self.chunk_len
            if task_ids is not None:
                task_ids = task_ids.repeat_interleave(n_chunks)   
            if trj_ids is not None: 
                trj_ids = trj_ids.repeat_interleave(n_chunks)
            if trj_seeds is not None: 
                trj_seeds = trj_seeds.repeat_interleave(n_chunks)
            timesteps = timesteps.reshape(-1, self.chunk_len)
            
        idx_precalc, distances_precalc = None, None
        if precalc_dict is not None: 
            # extract precalc_idx, precalc_distances
            idx_precalc = precalc_dict["idx"][self.precalc_counter: self.precalc_counter + cache_query.shape[0]]
            distances_precalc = precalc_dict["distances"][self.precalc_counter: self.precalc_counter + cache_query.shape[0]]
            self.precalc_counter += cache_query.shape[0]
            
        # retrieve
        distances_retrieved, idx_retrieved, vals_retrieved, _ = index.query_cache(
            cache_query, 
            k=self.top_k_ret,
            reshape_context=self.learnable_ret,
            compute_normed_distances=self.rel_ret_fraction,
            task_id=task_ids,
            trj_id=trj_ids,
            p_mask=p_mask,
            timesteps=timesteps, 
            idx_precalc=idx_precalc,
            distances_precalc=distances_precalc,
            # check how expensive max is here
            total_returns=total_returns,
            trj_seed=trj_seeds
        )
        if self.chunk_len is not None: 
            # vals_retrieved contains val like [batch_size * n_chunks, top_k * (self.cache_len + self.future_cache_len), ...]
            # i.e., increased batch
            # reshape to [batch_size, n_chunks, top_k * (self.cache_len + self.future_cache_len), ...]
            n_chunks = states.shape[1] // self.chunk_len
            vals_retrieved = {k: v.reshape(-1, n_chunks, *v.shape[1:]) for k, v in vals_retrieved.items()}

        return distances_retrieved, idx_retrieved, vals_retrieved 
    
    def get_action_pred(self, policy, states, actions, rewards, returns_to_go, timesteps, attention_mask,
                        deterministic, prompt,  is_eval=False, task_id=None, env_act_dim=None):
        context_trjs = None
        if self.learnable_ret:
            if self.sep_eval_cache and (self.eval_cache is None or not self.eval_cache.is_ready):
                pass
            else: 
                # retrieve every eval_ret_steps, otherwise keep context
                if self.eval_ret_steps == 1 or (timesteps[0][-1] % self.eval_ret_steps == 0):     
                    # retrieve either from cache or from eval_cache
                    index = self.eval_cache if self.sep_eval_cache else self.cache 
                    # retrieve relevant context for cross attention
                    _, _, context_trjs = self.retrieve_context_trjs(
                        index, states, actions, rewards, returns_to_go,
                        timesteps, attention_mask, prompt, task_ids=task_id,
                        p_mask=self.p_context_mask,
                    )
                    if self.eval_ret_steps > 1: 
                        # preserve context trjs
                        self.eval_ret_context_trjs = context_trjs
                else: 
                    context_trjs = self.eval_ret_context_trjs
        
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            policy_output = policy(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=True,
                deterministic=deterministic,
                prompt=prompt,
                task_id=task_id,
                ddp_kwargs=self.ddp_kwargs,
                context_trjs=context_trjs,
                use_inference_cache=self.use_inference_cache,
                past_key_values=self.past_key_values # None by default
            )

        if not is_eval and self.num_timesteps % 10000 == 0 and self.log_attn_maps:
            self._record_attention_maps(policy_output.attentions, step=self.num_timesteps, prefix="rollout")
            if policy_output.cross_attentions is not None:
                self._record_attention_maps(policy_output.cross_attentions, step=self.num_timesteps,
                                            prefix="rollout_cross", lower_triu=False)

        action_predicted = policy_output.action_preds[0, -1]
        if env_act_dim is not None:
            action_predicted = action_predicted[:env_act_dim]
        
        if not self.learnable_ret: 
            action_predicted = self.mix_action(policy, states, action_predicted, policy_output, attention_mask)
        if self.use_inference_cache:
            self.past_key_values = policy_output.past_key_values
        self.pred_counter += 1
        return action_predicted, action_predicted
    
    def mix_action(self, policy, states, action, policy_output, attention_mask):
        latent_features = policy_output.last_hidden_state[0, -1]
        # query cache
        cache_query = self.extract_representation(
            policy_output, policy, states=states, attention_mask=attention_mask, chunk_len=self.chunk_len
        ).detach().cpu().numpy()
        distances, _, vals_retrieved, _ = self.cache.query_cache(
            cache_query, k=self.top_k_ret, 
            reshape_context=self.learnable_ret, 
            compute_normed_distances=self.rel_ret_fraction
        )
        actions_retrieved = vals_retrieved["actions"]

        # prepare final action
        action = self.prepare_action(action, actions_retrieved, distances,
                                                features=latent_features, state=states[0, -1])
        return action

    def prepare_action(self, action_predicted, actions_retrieved, distances, features=None, state=None):
        actions_retrieved = actions_retrieved.to(self.device)
        if self.action_kind == "original":
            return action_predicted
        elif self.action_kind == "random":
            all_actions = torch.cat(
                [action_predicted.unsqueeze(0), actions_retrieved], dim=0)
            return all_actions[torch.randint(len(all_actions), (1,))].squeeze(0)
        elif "value" in self.action_kind:
            assert features is not None, "Need to provide latent features"
            # extract q-values
            all_actions = torch.cat(
                [action_predicted.unsqueeze(0), actions_retrieved], dim=0)
            q_vals = self.critic.q1_forward(all_actions, features=features.repeat(all_actions.shape[0], 1),
                                            obs=state.repeat(all_actions.shape[0], 1))
            if self.action_kind == "value_mixed": 
                q_vals = q_vals[1:]
                rel_q_vals = q_vals / (q_vals.sum() + 1e-8)
                actions_retrieved = (actions_retrieved * rel_q_vals).sum(dim=0)
                return (1 - self.ret_fraction) * action_predicted + self.ret_fraction * actions_retrieved 
            return all_actions[torch.argmax(q_vals)]
        elif self.top_k_ret > 1:
            if self.action_kind == "same_weight": 
                actions_retrieved = actions_retrieved.mean(dim=0)
                return (1 - self.ret_fraction) * action_predicted + self.ret_fraction * actions_retrieved
            distances = torch.from_numpy(distances).to(self.device)
            if self.cache.index.metric_type == 0:
                # inner product
                rel_distances = distances / (distances.sum() + 1e-8)
            else:
                # l2
                inv_distances = 1 / (distances + 1e-8)
                rel_distances = inv_distances / (inv_distances.sum() + 1e-8)
            actions_retrieved = (actions_retrieved * rel_distances.T).sum(dim=0)
            
            return (1 - self.ret_fraction) * action_predicted + self.ret_fraction * actions_retrieved 
        
        return (1 - self.ret_fraction) * action_predicted + self.ret_fraction * actions_retrieved[0]
    
    def update_eval_cache(self, states, actions, rewards, timesteps, is_first_episode): 
        if self.demo_tasks is not None and self.eval_task in self.demo_tasks:
            # print("Total return:", rewards.sum())
            return 
        if is_first_episode: 
            self.reset_eval_cache()  
        if not self.learnable_ret and not self.sep_eval_cache: 
            return
        # recompute actually rtgs
        returns_to_go = discount_cumsum_torch(rewards, 1).unsqueeze(0)
        total_return = returns_to_go[0][0].detach().cpu()
        if self.eval_cache is not None and not self.eval_cache.should_store_trj(total_return.item(), self.reward_scale):
            return
        keys, values = self.construct_keys_and_values_from_trj(states, actions, rewards, returns_to_go,
                                                               timesteps, total_return)
        if self.eval_cache is not None and self.eval_cache.is_ready: 
            self.eval_cache.add_to_cache(keys, values)            
        else: 
            self._setup_eval_cache(keys, values, total_return.item())
    
    @torch.no_grad()
    def construct_keys_and_values_from_trj(self, states, actions, rewards, returns_to_go, timesteps, total_return):
        """
        Steps: 
          - iterate each step in the trajectory (batched)
          - pad trajectory to max len
          - forward pass + extract representation
          - add to eval cache --> key + values of desired length 

        Args:
            states: torch.Tensor
            actions: torch.Tensor
            rewards: torch.Tensor
            returns_to_go: torch.Tensor
            timesteps: torch.Tensor
        """
        keys, values = [], collections.defaultdict(list)
        context_len = self.cache.context_len
        # extract all sequences + pad
        all_states, all_actions, all_rewards, all_returns_to_go, all_timesteps, all_masks = [], [], [], [], [], []
        returns_to_go, timesteps = returns_to_go.squeeze(0), timesteps.squeeze(0)
        
        if len(states.shape) == 4:
            # img encoder is same for policy and retriever
            states = self._precompute_image_embeds(states)
        
        seq_len = states.shape[0]
        for i in range(0, states.shape[0], self.cache_steps): 
            # construct input sequences 
            start, end = max(0, i - context_len), max(2, i)
            if self.rand_first_chunk and end < self.cache_steps:
                end = np.random.randint(2, min(self.cache_steps, seq_len))
            obs_shape, act_dim = states.shape[1:], actions.shape[-1]
            s, a, rtg, t, mask, r = self.pad_inputs(
                states[start: end].reshape(1, -1, *obs_shape), 
                actions[start: end].reshape(1, -1, act_dim),
                returns_to_go[start: end].reshape(1, -1, 1),
                timesteps[start: end].reshape(1, -1),
                rewards=rewards[start: end].reshape(1, -1, 1),
                context_len=context_len
            )
            if self.state_mean is not None and self.state_std is not None:
                s = (s - self.state_mean) / self.state_std
                
            all_states.append(s)
            all_actions.append(a)
            all_rewards.append(r)
            all_returns_to_go.append(rtg)
            all_timesteps.append(t)
            all_masks.append(mask)
        
            # extract "past" + "future" context
            start_idx, end_idx = compute_start_end_context_idx(i, states.shape[0], 
                                                               self.cache_len, self.future_cache_len,
                                                               full_context_len=self.full_context_len,
                                                               dynamic_len=self.dynamic_context_len)
            vals = self.construct_values_from_trj(states, actions, rewards, returns_to_go, timesteps,
                                                  total_return, start_idx, end_idx,
                                                  self.cache_len + self.future_cache_len)
            for k, v in vals.items():
                values[k].append(v)
        
        # concat input sequences
        all_states, all_actions, all_rewards, all_returns_to_go, all_timesteps, all_masks = torch.concat(all_states, dim=0), \
            torch.concat(all_actions, dim=0), torch.concat(all_rewards, dim=0), \
            torch.concat(all_returns_to_go, dim=0), torch.concat(all_timesteps, dim=0), torch.concat(all_masks, dim=0)
        
        embed_model = self.retriever if self.retriever is not None else self.policy
        for i in range(0, all_states.shape[0], self.batch_size): 
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                policy_output = embed_model(
                    states=all_states[i: i + self.batch_size],
                    actions=all_actions[i: i + self.batch_size],
                    rewards=all_rewards[i: i + self.batch_size],
                    returns_to_go=all_returns_to_go[i: i + self.batch_size],
                    timesteps=all_timesteps[i: i + self.batch_size].long(),
                    attention_mask=all_masks[i: i + self.batch_size],
                    return_dict=True,
                    with_log_probs=self.stochastic_policy,
                    deterministic=False,
                    prompt=None,
                    ddp_kwargs=self.ddp_kwargs,
                )
            key = self.extract_representation(policy_output, embed_model, states=s,
                                              attention_mask=all_masks[i: i + self.batch_size])
            keys.append(key.detach().cpu())
        
        for k, v in values.items():
            values[k] = torch.concat(v, dim=0)
        keys = torch.concat(keys, dim=0).numpy()
        if len(keys.shape) > 2:
            keys = keys.squeeze(1)
        return keys, values
    
    def construct_values_from_trj(self, states, actions, rewards, returns_to_go, timesteps,
                                  total_return, start_idx, end_idx, context_len):
        # ensure that all context tjrs are of same length (context_len), even for shorter epsiodes
        s, a, rtg, t, mask, r = self.pad_inputs(
            states[start_idx: end_idx].float().unsqueeze(0), 
            actions[start_idx: end_idx].unsqueeze(0),
            returns_to_go[start_idx: end_idx].reshape(1, -1, 1).float(),
            timesteps[start_idx: end_idx].unsqueeze(0),
            rewards=rewards[start_idx: end_idx].reshape(1, -1, 1),
            context_len=context_len
        )
        a = a.long() if self.policy.is_discrete else a
        return {"states": s.detach().cpu(), "actions": a.detach().cpu(), "returns_to_go": rtg.detach().cpu(), 
                "timesteps": t.detach().cpu(), "attention_mask": mask.detach().cpu(), "rewards": r.detach().cpu(),
                "total_return": total_return.unsqueeze(0)}
        
    def _record_retrieval_stats(self, context_trjs, task_ids, trj_ids, trj_seeds, total_returns, observations): 
        metrics = {}
        # log task id accuracy
        task_ids_ret = context_trjs["task_ids"].flatten(1)
        task_ids_rep = task_ids.cpu().repeat_interleave(task_ids_ret.shape[-1])        
        metrics["ret_task_acc"] = torchmetrics.functional.classification.accuracy(
            task_ids_ret.flatten(), task_ids_rep, 
            task="multiclass", num_classes=len(task_ids.unique()),
            validate_args=False
        ).item()
        # log trj id accuracy
        trj_ids_ret = context_trjs["trj_ids"].flatten(1)
        trj_ids_rep = trj_ids.cpu().repeat_interleave(trj_ids_ret.shape[-1])
        metrics["ret_trj_acc"] = torchmetrics.functional.classification.accuracy(
            trj_ids_ret.flatten(), trj_ids_rep, 
            task="multiclass", num_classes=len(trj_ids.unique()),
            validate_args=False
        ).item()
        # log trj seed accuracy
        if "trj_seeds" in context_trjs:
            trj_seed_ret = context_trjs["trj_seeds"].flatten(1)
            trj_seed_rep = trj_seeds.cpu().repeat_interleave(trj_seed_ret.shape[-1])
            metrics["ret_seed_acc"] = torchmetrics.functional.classification.accuracy(
                trj_seed_ret.flatten(), trj_seed_rep, 
                task="multiclass", num_classes=len(trj_seeds.unique()),
                validate_args=False
            ).item()
        # log abs_return_difference
        total_returns_ret = context_trjs["total_return"].flatten(1)
        if total_returns is not None: 
            total_returns = total_returns.cpu().repeat_interleave(total_returns_ret.shape[-1])
            metrics["trj_return_mae"] = torchmetrics.functional.regression.mean_absolute_error(
                total_returns_ret.flatten() * self.reward_scale, total_returns * self.reward_scale,
            ).item()
        return metrics
        
    def reset_eval_cache(self):
        if self.demo_tasks is not None and self.eval_task in self.demo_tasks:
            return 
        self._eval_cache = None
        
    def set_current_eval_task(self, name): 
        self.eval_task = name
        
    def _precompute_image_embeds(self, states): 
        states = states / 255.0
        img_embeds = []
        for i in range(0, states.shape[0], self.batch_size):
            with torch.no_grad():
                # unsqueeze first to add batch dimension of 1 --> recognized as image batch
                img_embeds.append(
                    self.policy.get_state_embeddings(states.unsqueeze(0)[:, i: i + self.batch_size]).squeeze(0)
                )
        return torch.concat(img_embeds, axis=0)
    
    def _dump_logs(self): 
        cache_stats = self.cache._get_buffer_stats(prefix="cache", midfix="c_")
        for k, v in cache_stats.items():
            self.logger.record(k, round(v, 2))
        super()._dump_logs()

    def _record_param_count(self):
        super()._record_param_count()
        if self.learnable_ret:
            counts = get_param_count(self.retriever, "retriever")
            for k, v in counts.items():
                self.logger.record(f"param_counts/{k}", v)
                
    def _get_torch_save_params(self):
        state_dicts, torch_vars = super()._get_torch_save_params()
        if self.learnable_ret: 
            state_dicts += ["retriever"]
        return state_dicts, torch_vars

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["cache", "eval_cache"]
    
    def load_retriever_weights(self, path):
        print(f"Loading retriever weights from: {path}")
        custom_objects = {"observation_space": None, "action_space": None}
        _, params, _ = load_from_zip_file(path, device=self.device, custom_objects=custom_objects)
        retriever_dict = params["retriever"]
        # models may be saved with "module." prefix, replace 
        retriever_dict = {k.replace("module.", "", 1): v for k, v in retriever_dict.items()}
        from_compiled_model = False
        if list(retriever_dict.keys())[0].startswith("_orig_mod."):
            print("Loaded retriever weights start with _orig_mod. From compiled model.")
            from_compiled_model = True
        else: 
            print("Loaded retriever weights are from uncompiled model.")        
        if not self.compile: 
            # remove compilation prefix
            retriever_dict = {k.replace("_orig_mod.", "", 1): v for k, v in retriever_dict.items()}
        elif self.compile and not from_compiled_model:
            # add compilation prefix
            retriever_dict = {f"_orig_mod.{k}": v for k, v in retriever_dict.items()}
        missing_keys, unexpected_keys = self.retriever.load_state_dict(retriever_dict, strict=False)
        if missing_keys:
            print("Missing key(s) in state_dict:", missing_keys)
        if unexpected_keys:
            print("Unexpected key(s) in state_dict:", unexpected_keys)

    def compute_target_return_val(self, env=None, task_id=0):
        target = super().compute_target_return_val(env, task_id)
        if self.sep_eval_cache and self.eval_cache is not None and self.target_return_type == "predefined_max_cache":
            new_target = self.eval_cache.values["total_return"].max().item()
            if new_target > 0: 
                target = round(new_target * 1.2) 
                self._last_target_return = target / self.reward_scale
        return target


class DiscreteCacheDecisionTransformerSb3(CacheDecisionTransformerSb3, DiscreteDecisionTransformerSb3):

    def get_action_pred(self, policy, states, actions, rewards, returns_to_go, timesteps, attention_mask,
                        deterministic, prompt, is_eval=False, task_id=None, env_act_dim=None):        
        inputs = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
            "return_dict": True,
            "deterministic": deterministic,
            "prompt": prompt,
            "task_id": task_id,
            "ddp_kwargs": self.ddp_kwargs
        }
        if self.learnable_ret:
            if self.sep_eval_cache and (self.eval_cache is None or not self.eval_cache.is_ready):
                pass
            else: 
                # retrieve every eval_ret_steps, otherwise keep context
                if self.eval_ret_steps == 1 or (timesteps[0][-1] % self.eval_ret_steps == 0):     
                    # retrieve either from cache or from eval_cache
                    index = self.eval_cache if self.sep_eval_cache else self.cache 
                    # retrieve relevant context for cross attention
                    _, _, context_trjs = self.retrieve_context_trjs(
                        index, states, actions, rewards, returns_to_go,
                        timesteps, attention_mask, prompt, task_ids=task_id,
                    )
                    inputs["context_trjs"] = context_trjs
                    if self.eval_ret_steps > 1: 
                        # preserve context trjs
                        self.eval_ret_context_trjs = context_trjs
                else: 
                    inputs["context_trjs"] = self.eval_ret_context_trjs
        
        if self.use_inference_cache: 
            # add only after context retrieval - as retriever is different LM
            inputs["past_key_values"] = self.past_key_values       
            inputs["use_inference_cache"] = self.use_inference_cache
            
        # expert-action inference mechanism
        if self.target_return_type == "infer":
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                policy_output = policy(**inputs)
            return_logits = policy_output.return_preds[:, -1]
            return_sample = policy.sample_from_rtg_logits(return_logits, **self.rtg_sample_kwargs)
            inputs["returns_to_go"][0, -1] = return_sample

        if not self.policy.tok_a_target_only: 
            # autoregressive action prediction
            # e.g., for discretizes continuous action space need to predict each action dim after another
            act_dim = actions.shape[-1] if env_act_dim is None else env_act_dim
            for i in range(act_dim):
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    policy_output = policy(**inputs)
                if not is_eval and self.num_timesteps % 10000 == 0 and self.log_attn_maps:
                    self._record_attention_maps(
                        policy_output.attentions, step=self.num_timesteps, prefix="rollout")
                    if policy_output.cross_attentions is not None:
                        self._record_attention_maps(policy_output.cross_attentions, step=self.num_timesteps + i,
                                                    prefix="rollout_cross", lower_triu=False)
                if self.a_sample_kwargs is not None: 
                    action_logits = policy_output.action_logits[0, -1, i]
                    inputs["actions"][0, -1, i] = sample_from_logits(action_logits, **self.a_sample_kwargs)
                else: 
                    inputs["actions"][0, -1, i] = policy_output.action_preds[0, -1, i]

                if i == 0 and not self.learnable_ret:
                    # extract hidden state prior to first action token
                    cache_query = self.extract_representation(
                        policy_output, policy, states=states, attention_mask=attention_mask, chunk_len=self.chunk_len
                    ).detach().cpu().numpy()
                    if len(cache_query.shape) > 2:
                        cache_query = cache_query.squeeze(1)
                if self.use_inference_cache: 
                    self.past_key_values = policy_output.past_key_values
                    inputs["past_key_values"] = self.past_key_values
                    
            action = inputs["actions"][0, -1]
        else: 
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    policy_output = policy(**inputs)
            action = policy_output.action_preds[0, -1]
            if self.use_inference_cache: 
                self.past_key_values = policy_output.past_key_values
            
        if not self.learnable_ret: 
            action = self.mix_action(states, action, policy_output, cache_query)
        if env_act_dim is not None: 
            action = action[:env_act_dim]
        return action, inputs["returns_to_go"][0, -1] if self.target_return_type == "infer" else action
    
    def mix_action(self, states, action, policy_output, cache_query):
        action_probs = self.extract_action_probs(policy_output)
        # retrieval augmentation
        distances, _, vals_retrieved, _ = self.cache.query_cache(
            cache_query, k=self.top_k_ret,
            reshape_context=self.learnable_ret,
            compute_normed_distances=self.rel_ret_fraction
        )
        actions_retrieved, states_retrieved = vals_retrieved["actions"], vals_retrieved["states"]

        if self.plot_ret_freq is not None and self.pred_counter % self.plot_ret_freq == 0:
            self.plot_retrieved_states(states[0, -1], action, states_retrieved, actions_retrieved)
        self.pred_counter += 1

        actions_retrieved_probs = self.convert_ret_actions_to_probs(
            actions_retrieved, n_actions=action_probs.shape[-1])
        action = self.prepare_action(action_probs, actions_retrieved_probs, distances)
        return action

    def extract_action_probs(self, policy_output):
        # extract last hidden state prior to the last action token
        action_logits = policy_output.action_logits[0, -1]
        if len(action_logits.shape) > 1:
            action_logits = action_logits.squeeze(0)
        return torch.softmax(action_logits, dim=-1)

    def convert_ret_actions_to_probs(self, actions_retrieved, n_actions):
        return torch.nn.functional.one_hot(actions_retrieved.flatten(), n_actions).float()

    def prepare_action(self, action_predicted, actions_retrieved, distances): 
        action_probs = super().prepare_action(action_predicted, actions_retrieved, distances)        
        return torch.argmax(action_probs).unsqueeze(0)

    def plot_retrieved_states(self, state, action, states_retrieved, actions_retrieved):
        fig = make_retrieved_states_plot(state, action, states_retrieved, actions_retrieved, self.pred_counter)
        self.logger.record("retrieved_states", Figure(fig, True), exclude="stdout")
        self.logger.dump(step=self.pred_counter)

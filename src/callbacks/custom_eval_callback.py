import os
import collections
import scipy.stats
import numpy as np
import torch
import wandb
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization, DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat
from .evaluation import custom_evaluate_policy
from ..envs.env_utils import extract_env_name
from ..envs.hn_scores import get_human_normalized_score, ENVID_TO_HNS
from ..envs.dn_scores import get_data_normalized_score, ENVID_TO_DNS
from ..envs.env_names import PROCGEN_ENVS


class CustomEvalCallback(EvalCallback):
    """
    We just want to swap evaluate_policy to custom_evalutate_policy, which works with DT.

    """
    def __init__(self, eval_env, first_step=True, log_eval_trj=False, prefix="eval", use_wandb=False, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.first_step = first_step
        self.log_eval_trj = log_eval_trj
        self.prefix = prefix
        # for some reason, steps in wandb need to be monotonically increasing: if we want to log "earlier" steps
        # define custom metric: https://github.com/wandb/wandb/issues/6554
        self.use_wandb = use_wandb
        self.wandb_is_defined = False
        if self.use_wandb and wandb.run is not None: 
            self.step_metric = f"{prefix}_step"
            wandb.define_metric(self.step_metric)

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            with torch.no_grad():
                episode_rewards, episode_lengths = custom_evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"{self.prefix}/mean_reward", float(mean_reward))
            self.logger.record(f"{self.prefix}/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record(f"{self.prefix}/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)
            
            # log human normalized scores
            env_name = extract_env_name(self.eval_env)
            if env_name in ENVID_TO_HNS: 
                self._log_normalized_scores(env_name, np.array(episode_rewards), "hns")
            elif env_name in ENVID_TO_DNS:
                self._log_normalized_scores(env_name, np.array(episode_rewards), "dns")

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()
            
            if self.log_eval_trj: 
                self._log_eval_trj(episode_rewards)
                
            self.wandb_is_defined = False
        if hasattr(self.model, "ddp") and self.model.ddp:
            torch.distributed.barrier()
        return continue_training

    def _on_training_start(self):
        if self.first_step:
            # Do an initial evaluation before training
            print("Initial evaluation...")
            self._on_step()
    
    def _log_normalized_scores(self, env_name, episode_rewards, score_type="hns"):
        score_fn = get_human_normalized_score if score_type == "hns" else get_data_normalized_score
        score = np.mean(score_fn(env_name, episode_rewards))
        if self.verbose > 0:
            print(f"{score_type}: {score:.2f}%")
        self.logger.record(f"{self.prefix}/{env_name}/{score_type}", score)
        
    def associate_wandb_metrics(self, metrics):
        # should only be done the very first time
        for metric in metrics:
            wandb.define_metric(metric, step_metric=self.step_metric)
            
    def _log_eval_trj(self, vals, key="reward_trj"):
        for i, ep_reward in enumerate(vals):
            if self.use_wandb and wandb.run is not None: 
                if not self.wandb_is_defined:
                    self.associate_wandb_metrics([f"{self.prefix}/{key}"])
                wandb.log({
                    f"{self.prefix}/{key}": ep_reward, self.step_metric: self.num_timesteps + i
                })
            else: 
                self.logger.record(f"{self.prefix}/{key}", ep_reward, exclude="stdout")
                self.logger.dump(self.num_timesteps + i)
        

class MultiEnvEvalCallback(EvalCallback):

    def __init__(self, eval_env, first_step=True, log_eval_trj=False, prefix="eval", use_wandb=False, **kwargs):
        super().__init__(eval_env, **kwargs)
        # dicts for keeping track of performance immediately after task was trained on
        # and performance after all tasks for measuring forgetting
        self.task_to_last_scores = collections.defaultdict(float)
        self.task_to_task_scores = collections.defaultdict(float)
        self.first_step = first_step
        self.log_eval_trj = log_eval_trj
        self.prefix = prefix
        self.use_wandb = use_wandb  
        self.wandb_is_defined = False      
        if self.use_wandb and wandb.run is not None: 
            self.step_metric = f"{prefix}_step"
            wandb.define_metric(self.step_metric)

    def _init_callback(self):
        super()._init_callback()
        # configure Logger to display more than 36 characters --> this kills runs due to duplicate keys
        # increase max_length for now.
        for format in self.logger.output_formats:
            if isinstance(format, HumanOutputFormat):
                format.max_length = 96

    def _on_step(self) -> bool:
        continue_training = True
        eval_fn = custom_evaluate_policy
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            avg_successes, avg_rewards, avg_hns, avg_dns,  = [], [], [], [],
            avg_episode_rewards, avg_episode_successes, avg_episode_dns, avg_episode_hns = collections.defaultdict(list), \
                collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
            num_envs = len(self.eval_env.envs)
            for idx, env in enumerate(self.eval_env.envs):
                env_name = extract_env_name(env, idx)
                env_id = f"{env_name}_{idx}"
                # Sync training and eval env if there is VecNormalize
                if self.model.get_vec_normalize_env() is not None:
                    try:
                        sync_envs_normalization(self.training_env, env)
                    except AttributeError:
                        raise AssertionError(
                            "Training and eval env are not wrapped the same way, "
                            "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                            "and warning above."
                        )

                if not isinstance(env, (DummyVecEnv, SubprocVecEnv)) and env_name not in PROCGEN_ENVS:
                    env = DummyVecEnv([lambda: env])
                if isinstance(self.eval_env, VecVideoRecorder):
                    # wrap the env with same recorder
                    env = VecVideoRecorder(env, video_folder=self.eval_env.video_folder, 
                                           record_video_trigger=self.eval_env.record_video_trigger,
                                           video_length=self.eval_env.video_length, 
                                           name_prefix=f"{self.prefix}_{env_name}")
                    env.name = env_name
                
                # Reset success rate buffer
                self._is_success_buffer = []

                with torch.no_grad():
                    episode_rewards, episode_lengths = eval_fn(
                        self.model,
                        env,
                        n_eval_episodes=self.n_eval_episodes,
                        render=self.render,
                        deterministic=self.deterministic,
                        return_episode_rewards=True,
                        warn=self.warn,
                        callback=self._log_success_callback,
                        task_id=idx
                    )

                if self.log_path is not None:
                    self.evaluations_timesteps.append(self.num_timesteps)
                    self.evaluations_results.append(episode_rewards)
                    self.evaluations_length.append(episode_lengths)

                    kwargs = {}
                    # Save success log if present
                    if len(self._is_success_buffer) > 0:
                        self.evaluations_successes.append(self._is_success_buffer)
                        kwargs = dict(successes=self.evaluations_successes)

                    np.savez(
                        self.log_path,
                        timesteps=self.evaluations_timesteps,
                        results=self.evaluations_results,
                        ep_lengths=self.evaluations_length,
                        **kwargs,
                    )

                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                self.last_mean_reward = mean_reward

                if self.verbose > 0:
                    print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                # Add to current Logger
                self.logger.record(f"{self.prefix}/{env_id}/mean_reward", float(mean_reward))
                self.logger.record(f"{self.prefix}/{env_id}/mean_ep_length", mean_ep_length)
                if num_envs == 1:
                    # redundant logging for easier aggregation
                    self.logger.record(f"{self.prefix}/mean_reward", float(mean_reward))
                    self.logger.record(f"{self.prefix}/mean_ep_length", mean_ep_length)
                elif idx == self.model.current_task_id:
                    self.logger.record(f"{self.prefix}/cur_task_mean_reward", float(mean_reward))
                avg_rewards.append(float(mean_reward))

                # log success rates
                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    avg_successes.append(success_rate)
                    self._log_success_rates(env_id, success_rate, num_envs=num_envs, idx=idx)
                
                # log human normalized scores
                if env_name in ENVID_TO_HNS: 
                    all_hns = get_human_normalized_score(env_name, np.array(episode_rewards))
                    hns = np.mean(all_hns)
                    avg_hns.append(hns)
                    self._log_normalized_scores(env_id, hns, num_envs=num_envs, idx=idx, score_type="hns")
                if env_name in ENVID_TO_DNS: 
                    all_dns = get_data_normalized_score(env_name, np.array(episode_rewards))
                    dns = np.mean(all_dns)
                    avg_dns.append(dns)
                    self._log_normalized_scores(env_id, dns, num_envs=num_envs, idx=idx, score_type="dns")

                # Dump log so the evaluation results are printed with the correct timestep
                self.logger.record(f"time/{env_id}/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(self.num_timesteps)
                
                if self.log_eval_trj: 
                    avg_episode_rewards[f"{self.prefix}/{env_id}/reward_trj"] = episode_rewards
                    if len(self._is_success_buffer) > 0:
                        avg_episode_successes[f"{self.prefix}/{env_id}/success_trj"] = self._is_success_buffer
                    if env_name in ENVID_TO_HNS: 
                        avg_episode_hns[f"{self.prefix}/{env_id}/hns_trj"] = all_hns
                    if env_name in ENVID_TO_DNS: 
                        avg_episode_dns[f"{self.prefix}/{env_id}/dns_trj"] = all_dns
                    
                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.best_mean_reward = mean_reward
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()

                # Trigger callback after every evaluation, if needed
                if self.callback is not None:
                    continue_training = continue_training and self._on_event()
                
                if hasattr(env, "video_recorder") and env.video_recorder is not None:
                    env.close_video_recorder()
                    del env.video_recorder

            self._log_forgetting_scores()

            # log avg success
            if len(avg_successes) > 1:
                avg_success_rate = np.mean(avg_successes)
                iqm_success_rate = scipy.stats.trim_mean(avg_successes, proportiontocut=0.25, axis=None)
                if self.verbose > 0:
                    print(f"Success rate: {100 * avg_success_rate:.2f}%")
                self.logger.record(f"{self.prefix}/avg_success_rate", avg_success_rate)
                self.logger.record(f"{self.prefix}/iqm_success_rate", iqm_success_rate)
                self.logger.dump(self.num_timesteps)
            if len(avg_rewards) > 1:
                avg_rewards = np.mean(avg_rewards)
                self.logger.record(f"{self.prefix}/avg_rewards", avg_rewards)
                self.logger.dump(self.num_timesteps)
            if len(avg_hns) > 1:
                avg_hns_score = np.mean(avg_hns)
                iqm_hns = scipy.stats.trim_mean(avg_hns, proportiontocut=0.25, axis=None)
                self.logger.record(f"{self.prefix}/avg_hns", avg_hns_score)
                self.logger.record(f"{self.prefix}/iqm_hns", iqm_hns)
                self.logger.dump(self.num_timesteps)
            if len(avg_dns) > 1:
                avg_dns_score = np.mean(avg_dns)
                iqm_dns = scipy.stats.trim_mean(avg_dns, proportiontocut=0.25, axis=None)
                self.logger.record(f"{self.prefix}/avg_dns", avg_dns_score)
                self.logger.record(f"{self.prefix}/iqm_dns", iqm_dns)
                self.logger.dump(self.num_timesteps)
                
            if self.log_eval_trj:
                avg_ep_reward_trj = np.stack(list(avg_episode_rewards.values())).mean(0)
                all_key_vals = avg_episode_rewards
                if len(self._is_success_buffer) > 0:
                    avg_episode_successes[f"{self.prefix}/avg_success_trj"] = np.stack(list(avg_episode_successes.values())).mean(0)
                    all_key_vals = {**all_key_vals, **avg_episode_successes}
                if len(avg_hns) > 0:
                    avg_episode_hns[f"{self.prefix}/avg_hns_trj"] = np.stack(list(avg_episode_hns.values())).mean(0)
                    all_key_vals = {**all_key_vals, **avg_episode_hns}
                if len(avg_dns) > 0:
                    avg_episode_dns[f"{self.prefix}/avg_dns_trj"] = np.stack(list(avg_episode_dns.values())).mean(0)
                    all_key_vals = {**all_key_vals, **avg_episode_dns}
                self._log_eval_trj(avg_ep_reward_trj, all_key_vals)
                
            self.wandb_is_defined = True
        if hasattr(self.model, "ddp") and self.model.ddp:
            torch.distributed.barrier()
        return continue_training

    def _log_success_callback(self, locals_, globals_) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        if isinstance(info, list):
            info = info[0]

        if locals_["done"]:
            maybe_is_success = info.get("is_success") or info.get("success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_training_end(self) -> None:
        self._log_forgetting_scores()

    def _log_forgetting_scores(self):
        if len(self.task_to_task_scores) > 1 and len(self.task_to_last_scores) > 1:
            forgetting_scores = []
            for task_id, task_score in self.task_to_task_scores.items():
                forgetting = task_score - self.task_to_last_scores[task_id]
                self.logger.record(f"{self.prefix}/{task_id}/forgetting", float(forgetting))
                forgetting_scores.append(forgetting)
            self.logger.record(f"{self.prefix}/forgetting", np.mean(forgetting_scores))
            self.logger.dump(self.num_timesteps)

    def _on_training_start(self):
        if self.first_step:
            # Do an initial evaluation before training
            print("Initial evaluation...")
            self._on_step()
            
    def _log_success_rates(self, env_id, success_rate, num_envs=1, idx=0):
        if self.verbose > 0:
            print(f"Success rate: {100 * success_rate:.2f}%")
        self.logger.record(f"{self.prefix}/{env_id}/success_rate", success_rate)
        if num_envs == 1:
            # redundant logging for easier aggregation
            self.logger.record(f"{self.prefix}/success_rate", success_rate)
        else:
            self.task_to_last_scores[env_id] = success_rate
            if idx == self.model.current_task_id:
                self.logger.record(f"{self.prefix}/cur_task_success_rate", success_rate)
                self.task_to_task_scores[env_id] = success_rate

    def _log_normalized_scores(self, env_id, score, num_envs, idx=0, score_type="hns"):
        if self.verbose > 0:
            print(f"{score_type}: {score:.2f}%")
        self.logger.record(f"{self.prefix}/{env_id}/{score_type}", score)
        if num_envs == 1:
            # redundant logging for easier aggregation
            self.logger.record(f"{self.prefix}/{score_type}", score)
        else:
            self.task_to_last_scores[env_id] = score
            if idx == self.model.current_task_id:
                self.logger.record(f"{self.prefix}/cur_task_{score_type}", score)
                self.task_to_task_scores[env_id] = score
        return score
    
    def _log_eval_trj(self, vals, key_val_dict, key="avg_reward_trj"):
        # dump
        self.logger.dump(self.num_timesteps)
        # for some tensorboard-specific reason, all trj rewards need to be written together. 
        for ep_idx, ep_reward in enumerate(vals):
            if self.use_wandb and wandb.run is not None: 
                # use different "step_metric" for tracking eval trj (wandb issues)
                if not self.wandb_is_defined:
                    self.associate_wandb_metrics(
                        [f"{self.prefix}/{key}"] + [k for k in key_val_dict.keys()]
                    )
                wandb.log({
                    f"{self.prefix}/{key}": ep_reward, 
                    **{k: v[ep_idx] for k, v in key_val_dict.items()},
                    self.step_metric: self.num_timesteps + ep_idx
                })
            else: 
                self.logger.record(f"{self.prefix}/{key}", ep_reward, exclude="stdout")
                # record individual env episode rewards
                for k, v in key_val_dict.items():
                    self.logger.record(k, v[ep_idx], exclude="stdout")
                self.logger.dump(self.num_timesteps + ep_idx)

    def associate_wandb_metrics(self, metrics):
        # should only be done the very first time
        for metric in metrics:
            wandb.define_metric(metric, step_metric=self.step_metric)
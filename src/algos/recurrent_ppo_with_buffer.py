import copy
import gym
import time
import sys
import numpy as np
import torch as th 
from pathlib import Path
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.save_util import save_to_pkl
from stable_baselines3.common.utils import safe_mean


class RecurrentPPOWithBuffer(RecurrentPPO):
    
    def __init__(self, policy, env, replay_buffer_size=100000, buffer_save_path=None, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.save_buffer = buffer_save_path is not None
        self.buffer_save_path = buffer_save_path
        if self.save_buffer:
            # needs to be the actual file name. 
            assert self.buffer_save_path is not None, "Please specify buffer save path."
            self.buffer_save_path = Path(self.buffer_save_path.replace(" ", ""))
            self.buffer_save_path.parent.mkdir(parents=True, exist_ok=True)
            self.replay_buffer = ReplayBuffer(
                    replay_buffer_size,
                    self.observation_space,
                    self.action_space,
                    device=self.device,
                    n_envs=self.n_envs,
            ) 

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = copy.deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts).float().to(self.device)
                actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :],
                            lstm_states.vf[1][:, idx : idx + 1, :],
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False]).float().to(self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            # store obs to replay buffer
            if self.save_buffer and self.num_timesteps <= self.total_timesteps: 
                self.replay_buffer.add(obs=self._last_obs.copy(), next_obs=new_obs.copy(), action=actions.copy(),
                                       reward=rewards.copy(), done=dones.copy(), infos=infos.copy())

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones).float().to(self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1,
        eval_env = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "RecurrentPPO",
        eval_log_path = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        self.total_timesteps = total_timesteps 
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                if len(self.ep_success_buffer) > 0:
                    self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))    
                
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()
        
        if self.save_buffer:
            # dump replay buffer
            print(f"Saving replay buffer to {self.buffer_save_path}")
            save_to_pkl(self.buffer_save_path, self.replay_buffer, True)
        
        return self
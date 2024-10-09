import time
import numpy as np
import omegaconf
from copy import deepcopy
from procgen import ProcgenEnv, ProcgenGym3Env
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.env_util import DummyVecEnv
from procgen.env import ToBaselinesVecEnv


class StickyBaselinesEnv(ToBaselinesVecEnv):
    """
    Sticky action wrapper for Procgen. 
    """
    def __init__(self, env, p_sticky: float = 0.1, **kwargs):
        super().__init__(env, **kwargs)
        self.p_sticky = p_sticky
        self.sticky_actions = np.zeros(self.num_envs, dtype=int)

    def reset(self):
        self.sticky_actions.fill(0)
        return super().reset()

    def step_async(self, ac):
        repeat_action_mask = np.random.rand(self.num_envs) < self.p_sticky
        self.sticky_actions[~repeat_action_mask] = ac[~repeat_action_mask]
        return super().step_async(self.sticky_actions)


def make_sticky_procgen_env(env_name, num_envs, p_sticky=0, **kwargs):
    return StickyBaselinesEnv(ProcgenGym3Env(num=num_envs, env_name=env_name, **kwargs), p_sticky=p_sticky)


def get_procgen_constructor(envid, distribution_mode="easy", time_limit=None, env_kwargs=None):
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    if isinstance(env_kwargs, omegaconf.DictConfig):
        env_kwargs = omegaconf.OmegaConf.to_container(env_kwargs, resolve=True, throw_on_missing=True)
    p_sticky = env_kwargs.pop("p_sticky", 0)
    num_envs = env_kwargs.pop("num_envs", 1)
    norm_reward = env_kwargs.pop("norm_reward", False)
    def make():
        if p_sticky > 0: 
            env = make_sticky_procgen_env(env_name=envid, num_envs=num_envs, distribution_mode=distribution_mode,
                                          p_sticky=p_sticky, **env_kwargs)
        else: 
            env = ProcgenEnv(env_name=envid, num_envs=num_envs,
                            distribution_mode=distribution_mode, **env_kwargs)
        # monitor to obtain ep_rew_mean, ep_rew_len + extract rgb images from dict states
        env = CustomVecMonitor(VecExtractDictObs(env, 'rgb'), time_limit=time_limit)
        env = VecTransposeImage(env)
        if norm_reward: 
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
        env.name = envid
        return env
    return make


class CustomVecMonitor(VecMonitor):
    """
    Custom version of VecMonitor that allows for a timelimit.
    Once, timelimit is hit, we also need to reset the environment.
    We can however, not save the reset state there. 
    """
    def __init__(
        self,
        venv,
        filename=None,
        info_keywords=(),
        time_limit=None
    ):
        super().__init__(venv, filename, info_keywords)
        self.time_limit = time_limit

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if self.time_limit is not None and self.episode_lengths[i] >= self.time_limit:
                dones[i] = True
                # send action -1 to reset ProcgenEnv: https://github.com/openai/procgen/issues/40#issuecomment-633720234
                if self.num_envs > 1:
                    raise NotImplementedError("Resetting ProcgenEnv with multiple environments is not supported.")
                self.venv.step(np.ones((1,), dtype=int) * -1)
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos


class CustomDummyVecEnv(DummyVecEnv):
    """
    Custom version of DummyVecEnv that allows wrapping ProcgenEnvs. 
    By default, ProcgenEnvs are vectorized already. 
    Therefore wrapping different tasks in a single DummyVecEnv fails, due to returning of vectorized infor buffers.
    """
    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                # self.buf_infos[env_idx]terminal_observation"] = obs
                self.buf_infos[env_idx][0]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

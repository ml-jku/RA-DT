import numpy as np
from .buffer_utils import discount_cumsum_np, compute_rtg_from_target


class Trajectory:

    def __init__(self, obs_shape, action_dim, max_len=1024, task_id=0, trj_id=0, trj_seed=0,
                 relative_pos_embds=False, handle_timeout_termination=True,
                 sample_full_seqs_only=False, last_seq_only=False, init_trj_buffers=True,
                 episodic=False):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.max_len = max_len
        self.relative_pos_embds = relative_pos_embds
        self.handle_timeout_termination = handle_timeout_termination
        self.sample_full_seqs_only = sample_full_seqs_only
        self.task_id = task_id
        self.trj_id = trj_id
        self.trj_seed = trj_seed
        self.last_seq_only = last_seq_only
        self.init_trj_buffers = init_trj_buffers
        self.episodic = episodic
        if self.init_trj_buffers: 
            assert obs_shape is not None and action_dim is not None, "obs_shape and action_dim must be provided"
            self.observations = np.zeros((self.max_len, ) + self.obs_shape, dtype=np.float32)
            self.next_observations = np.zeros((self.max_len, ) + self.obs_shape, dtype=np.float32)
            self.actions = np.zeros((self.max_len, self.action_dim), dtype=np.float32)
            self.rewards = np.zeros((self.max_len), dtype=np.float32)
            self.timesteps = np.zeros((self.max_len), dtype=np.float32)
            self.timeouts = np.zeros((self.max_len), dtype=np.float32)
        self.pos = 0
        self.full = False
        self.returns_to_go = None

    def add(self, obs, next_obs, action, reward, done, infos=None):
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.timesteps[self.pos] = np.array(self.pos).copy()
        if self.handle_timeout_termination:
            # we assume there is only one environment --> index 0 of infos
            self.timeouts[self.pos] = np.array(infos[0].get("TimeLimit.truncated", False))
        self.pos += 1
        if done:
            self.add_dones()
        if self.pos == self.max_len:
            self.full = True
        return self.full

    def add_full_trj(self, obs, next_obs, action, reward, done, task_id, trj_id, returns_to_go=None, trj_seed=None):
        self.pos = len(obs)
        if self.episodic:
            total_reward = np.sum(reward) 
            reward = np.zeros_like(reward)
            reward[-1] = total_reward
        if self.init_trj_buffers: 
            # buffers already, exist populate them with given trajectory
            self.observations[:self.pos] = obs
            if next_obs is not None:
                self.next_observations[:self.pos] = next_obs
            self.actions[:self.pos] = action
            self.rewards[:self.pos] = reward
            self.timesteps[:self.pos] = np.arange(0, self.pos)
        else: 
            # buffers do not exist, assign using the given trajectory
            self.observations = obs
            self.next_observations = next_obs
            self.actions = action
            self.rewards = reward
            self.timesteps = np.arange(0, self.pos)
            self.timeouts = np.zeros((self.pos), dtype=np.float32)
        self.add_dones(is_done=done[-1])
        self.full = True
        self.task_id = task_id
        self.trj_id = trj_id
        self.trj_seed = trj_seed
        self.returns_to_go = returns_to_go

    def sample(self, context_len=1, full_trj=False, end_idx=None):
        """
        Samples a trajectory from the buffer.

        It is important to sample the end_idx first. 
        Otherwise we may get trajectories that can't actually happen during evaluation:
        E.g., for context len=20, could have start_idx=5, end_idx=10 --> can't happen
        If we sampled end_idx first: end_idx=10 and start_idx=min(10 - context_len, 0), which could be a valid trj. 
        
        Args:
            context_len (int, optional): The length of the context to include in the trajectory. Defaults to 1.
            full_trj (bool, optional): Whether to sample the full trajectory. Defaults to False.
            end_idx (int, optional): The index of the end of the trajectory. Defaults to None.

        Returns:
            tuple: A tuple containing the sampled trajectory.
        """
        if full_trj: 
            start, end = 0, self.pos
        elif end_idx is not None:
            end = min(end_idx, self.pos)
            start = max(0, end - context_len)
        else: 
            # start = np.random.randint(0, self.pos, size=1)[0]
            # end = min(start + context_len, self.pos)
            end = np.random.randint(1, self.pos, size=1)[0] if self.pos > 1 else 1
            start = max(0, end - context_len)
        if self.last_seq_only and start < context_len:
            # ensure that, the agent also has the possibility to see the first n steps of a trajectory
            end = np.random.randint(start + 1,  min(start + context_len, self.pos))
        if self.sample_full_seqs_only and (end - start) < context_len:
            # ensure that only full sequences are sampled
            residual = context_len - (end - start)
            if context_len > self.pos:
                start = 0
                end = self.pos
            elif start - residual >= 0:
                start -= residual
            elif end + residual < self.pos:
                end += residual
        return self._get_samples(start, end)

    def _get_samples(self, start, end):
        timesteps = self.timesteps[start: end]
        dones = self.dones[start: end]
        if self.relative_pos_embds:
            timesteps = np.arange(len(timesteps))
        if self.handle_timeout_termination:
            dones = (dones * (1 - self.timeouts[start: end]))
        obs = self.observations[start: end]
        return obs, \
               self.next_observations[start: end] if self.next_observations is not None else np.zeros_like(obs), \
               self.actions[start: end], self.rewards[start: end], \
               self.returns_to_go[start: end], \
               timesteps, dones, self.task_id, self.trj_id, self.trj_seed

    def prune_trajectory(self):
        # to avoid OOM issues.
        self.observations = self.observations[:self.pos]
        self.next_observations = self.next_observations[:self.pos] if self.next_observations is not None else None
        self.actions = self.actions[:self.pos]
        self.rewards = self.rewards[:self.pos]
        self.timesteps = self.timesteps[:self.pos]
        self.timeouts = self.timeouts[:self.pos]

    def setup_final_trj(self, target_return=None, compute_stats=True):
        self.compute_returns_to_go(target_return=target_return)
        if compute_stats: 
            self.compute_mean_reward()
            self.compute_std_reward()
        self.prune_trajectory()

    def compute_returns_to_go(self, target_return=None):
        if self.returns_to_go is not None:
            # was already initialized when adding full trajectory
            return
        if target_return is not None:
            self.returns_to_go = compute_rtg_from_target(self.rewards, target_return)
            self.total_return = self.rewards.sum()
        else:
            self.returns_to_go = discount_cumsum_np(self.rewards[:self.pos], 1)
            self.total_return = self.returns_to_go[0]

    def compute_mean_reward(self):
        self.mean_reward = self.rewards[:self.pos].mean()

    def compute_std_reward(self):
        self.std_reward = self.rewards[:self.pos].std()

    def add_dones(self, is_done=True):
        self.dones = np.zeros(self.pos)
        if is_done:
            self.dones[-1] = 1

    def size(self):
        return self.pos

    def __len__(self):
        return self.pos

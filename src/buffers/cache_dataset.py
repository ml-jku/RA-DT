import collections
import torch
import numpy as np
from pathlib import Path
from .trajectory_dataset import TrajectoryDataset
from .buffer_utils import compute_start_end_context_idx


class CacheDataset(TrajectoryDataset):
    def __init__(self, trajectories, env, context_len, action_pad, cache_steps=1, rand_first_chunk=False, **kwargs):
        super().__init__(trajectories, env, context_len, action_pad, **kwargs)
        self.trj_idx_to_count = collections.defaultdict(int)
        self.cache_steps = cache_steps
        self.rand_first_chunk = rand_first_chunk

    def get_single_sample_from_memory(self, trj, full_trj=False, idx=None):
        assert idx is not None, "CacheDataset requires idx."
        # handle first step
        # does not make sense to sample the very first step of trajectory alone, as action not included there
        end_idx = max(self.trj_idx_to_count[idx], 2)
        self.trj_idx_to_count[idx] += self.cache_steps
        # self.trj_idx_to_count[idx] += self.cache_steps
        # end_idx = min(self.trj_idx_to_count[idx], len(trj))
        if self.rand_first_chunk and end_idx < self.cache_steps:
            end_idx = np.random.randint(2, min(len(trj), self.cache_steps))
        s, s1, a, r, togo, t, done, task_id, trj_id, trj_seed = trj.sample(self.context_len, end_idx=end_idx)
        return s, s1, a, r, togo, t, done, task_id, trj_id, None, trj_seed

    def get_single_sample_from_disk(self, path, idx):
        # directly load subset of trajectory from disk making use of trj_lengths
        end_idx = max(self.trj_idx_to_count[idx], 2)
        end_idx = min(end_idx, self.trj_lengths[path])
        self.trj_idx_to_count[idx] += self.cache_steps
        start_idx = max(0, end_idx - self.context_len)
        s, a, r, done_flag, togo = self.load_trj(path, start_idx=start_idx, end_idx=end_idx)
        assert togo is not None, "RTGs must be stored in trj file."
        r = r.astype(np.float32)
        if len(a.shape) == 1:
            a = np.expand_dims(a, -1)
        if isinstance(done_flag, (list, tuple, np.ndarray)):
            done_flag = done_flag[..., -1]
        done = np.zeros(len(s))
        done[-1] = done_flag
        s1 = np.zeros_like(s)
        t = np.arange(start_idx, end_idx)
        task_id, trj_id, trj_seed = 0, idx, 0
        return s, s1, a, r, togo, t, done, task_id, trj_id, None, trj_seed


class CacheWithContextDataset(CacheDataset):
    def __init__(self, trajectories, env, context_len, action_pad, cache_context_len, 
                 future_context_len, full_context_len=True, dynamic_context_len=False, **kwargs):
        super().__init__(trajectories, env, context_len, action_pad, **kwargs)
        self.cache_context_len = cache_context_len
        self.future_context_len = future_context_len
        self.full_context_len = full_context_len
        self.dynamic_context_len = dynamic_context_len
    
    def __getitem__(self, idx):
        s, a, s1, r, togo, t, mask, done, task_id, trj_id, action_mask, total_return, trj_seed = super().__getitem__(idx)
        # get context trjs 
        c_s, c_a, _, c_r, c_togo, c_t, c_mask, _, _ = self.extract_context_trjs(idx)
        return s, a, s1, r, togo, t, mask, done, task_id, trj_id, \
            action_mask, total_return, trj_seed, c_s, c_a, c_r, c_togo, c_t, c_mask,
    
    def extract_context_trjs(self, idx):
        trj = self.trajectories[idx]
        if isinstance(trj, (str, Path)):
            # load from disk
            path = str(trj)
            s, s1, a, r, togo, t, done, action_mask = self.get_context_from_disk(path, idx)
        else:
            # samples stored in memory, load from there
            s, s1, a, r, togo, t, done, action_mask = self.get_context_from_memory(trj, idx)
        
        # postprocess states, actions
        if len(s.shape) == 4 and self.to_rgb:
            # convert to "RGB" by repeating the gray-scale channel
            s = np.repeat(s, 3, axis=1)
            s1 = np.repeat(s1, 3, axis=1)
        if self.env is not None:
            s = self.env.normalize_obs(s)
            s1 = self.env.normalize_obs(s1)
            
        padding = max(0, (self.cache_context_len + self.future_context_len) - s.shape[0])
        mask = self.make_attention_mask(padding, s.shape[0])
        action_mask = np.ones_like(a, dtype=np.int32) if action_mask is None else action_mask
        if self.max_act_dim is not None and a.dtype.kind == "f":
            a, action_mask = self.pad_actions(a)
        if self.max_state_dim is not None and len(s.shape) == 2:
            s, s1 = self.pad_states(s, s1)
        if padding:
            s, s1, a, r, togo, t, done, action_mask = self.pad_sequences(s, s1, a, r, togo, t, 
                                                                         done, action_mask, padding)
        if len(s.shape) == 4 and self.transforms is not None:
            # perform image augmentations
            s = self.transforms(torch.from_numpy(s).float())
            s1 = self.transforms(torch.from_numpy(s1).float())
        
        return s, a, s1, np.expand_dims(r, axis=1), np.expand_dims(togo, axis=1), \
            t, mask, done, action_mask

    def get_context_from_memory(self, trj, idx):                         
        # return s, s1, a, r, togo, t, done, task_id, trj_id, None
        assert idx is not None, "CacheDataset requires idx."
        cur_trj_idx = self.trj_idx_to_count[idx] - self.cache_steps
        # cur_trj_idx = self.trj_idx_to_count[idx]
        start, end = compute_start_end_context_idx(cur_trj_idx, len(trj),
                                                   self.cache_context_len, self.future_context_len,
                                                   full_context_len=self.full_context_len,
                                                   dynamic_len=self.dynamic_context_len)
        s, s1, a, r, togo, t, done, _, _, _ = trj._get_samples(start, end)
        return s, s1, a, r, togo, t, done, None
    
    def get_context_from_disk(self, path, idx):
        # directly load subset of trajectory from disk making use of trj_lengths
        cur_trj_idx = self.trj_idx_to_count[idx] - self.cache_steps
        start, end = compute_start_end_context_idx(cur_trj_idx, self.trj_lengths[path], 
                                                   self.cache_context_len, self.future_context_len,
                                                   full_context_len=self.full_context_len,
                                                   dynamic_len=self.dynamic_context_len)
        s, a, r, done_flag, togo = self.load_trj(path, start_idx=start, end_idx=end)
        assert togo is not None, "RTGs must be stored in trj file."
        r = r.astype(np.float32)
        if len(a.shape) == 1:
            a = np.expand_dims(a, -1)
        if isinstance(done_flag, (list, tuple, np.ndarray)):
            done_flag = done_flag[..., -1]
        done = np.zeros(len(s))
        done[-1] = done_flag
        s1 = np.zeros_like(s)
        t = np.arange(start, end)
        return s, s1, a, r, togo, t, done, None

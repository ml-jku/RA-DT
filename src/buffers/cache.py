import collections
import functools
import faiss
import torch
import numpy as np
from shutil import rmtree
from pathlib import Path
from dataclasses import dataclass
from autofaiss import build_index
from torch.utils.data import DataLoader
from tqdm import tqdm
from .trajectory_buffer import TrajectoryReplayBuffer
from .trajectory import Trajectory
from .cache_dataset import CacheDataset, CacheWithContextDataset
from .samplers import TaskAwareIdSampler


def reset_folder_(p):
    path = Path(p)
    rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)


def faiss_read_index(path, mmap=False):
    if mmap: 
        return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    else: 
        return faiss.read_index(str(path))
        

def compute_nb_cores(nb_cores):
    if nb_cores is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            num_cores = torch.multiprocessing.cpu_count()
            nb_cores = num_cores // num_gpus
    elif nb_cores > torch.multiprocessing.cpu_count():
        nb_cores = torch.multiprocessing.cpu_count()
    return nb_cores


@dataclass
class TrajectoryReplayBufferSamplesWithContext:
    """We make this a dataclass (instead of NamedTuple TrajectoryReplayBufferSamples) for mutability."""
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    rewards_to_go: torch.Tensor
    timesteps: torch.Tensor
    attention_mask: torch.Tensor
    dones: torch.Tensor
    task_ids: torch.Tensor
    trj_ids: torch.Tensor
    action_mask: torch.Tensor
    total_returns: torch.Tensor
    trj_seeds: torch.Tensor
    context_observations: torch.Tensor
    context_actions: torch.Tensor
    context_rewards: torch.Tensor
    context_rewards_to_go: torch.Tensor
    context_timesteps: torch.Tensor
    context_attention_mask: torch.Tensor


class Cache(TrajectoryReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        cache_path=".tmp/.index",
        p_rand_trj=0,
        p_task_rand_trj=0,
        p_sample_task_rand_trj=0,
        p_null_doc=0,
        cache_steps=1,
        sim_cutoff_k=100,
        reweight_top_k=1,
        dist_weight=1, 
        return_weight=0, 
        task_weight=0, 
        pos_weight=0,
        improvement_weight=0,
        seed_weight=0, 
        p_rand_neighbour=0,
        p_ret_noise=0,
        p_ret_blend=0,
        percent_noise_std=0.1,
        noise_std=0,
        blend_alpha=0.5,
        n_blend=1, 
        deduplicate=False,
        norm=False,
        reweight_sample=False,
        exclude_same_trjs=False,
        only_same_task=False, 
        learnable_ret=False,
        full_context_len=False,
        dynamic_context_len=True,
        rand_first_chunk=False,
        reweight_mult=False,
        approx_sort=False,
        use_gpu=False,
        force_improvement=False,
        sim_cutoff=None,
        index_kwargs=None,
        min_return=None,
        sample_kind=None,
        top_k=None,
        cache_context_len=None,
        future_context_len=None,
        dedup_kwargs=None,
        nprobe=None,
        min_seq_len=None,
        min_return_for_ret=None, 
        n_rand_chunks=None,
        **kwargs,
    ):
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        self.norm = norm
        self.sample_kind = sample_kind
        self.p_rand_trj = p_rand_trj
        self.p_task_rand_trj = p_task_rand_trj
        self.p_sample_task_rand_trj = p_sample_task_rand_trj
        self.reweight_sample = reweight_sample
        self.min_return = min_return
        self.exclude_same_trjs = exclude_same_trjs
        self.only_same_task = only_same_task
        self.learnable_ret = learnable_ret
        self.cache_context_len = cache_context_len
        self.cache_steps = cache_steps
        self.future_context_len = future_context_len
        self.full_context_len = full_context_len
        self.dynamic_context_len = dynamic_context_len
        self.rand_first_chunk = rand_first_chunk
        self.p_null_doc = p_null_doc
        self.p_rand_neighbour = p_rand_neighbour
        self.use_gpu = use_gpu 
        self.force_improvement = force_improvement
        # typically k is passed by agent, in case of eval cache and reweighting needs to be passed
        self.top_k = top_k
        self.reweight_mult = reweight_mult
        self.reweight_top_k = reweight_top_k
        self.dist_weight = dist_weight
        self.task_weight = task_weight
        self.return_weight = return_weight
        self.pos_weight = pos_weight
        self.improvement_weight = improvement_weight
        self.seed_weight = seed_weight
        self.reweight = any([self.return_weight, self.task_weight, self.pos_weight,
                             self.improvement_weight, self.seed_weight])
        # no need to do full sort in case of reweighting (top-k)
        self.approx_sort = approx_sort if not self.reweight else True
        self.sim_cutoff_k = sim_cutoff_k
        self.sim_cutoff = sim_cutoff
        self.deduplicate = deduplicate
        self.nprobe = nprobe
        self.min_seq_len = min_seq_len
        self.min_return_for_ret = min_return_for_ret
        self.n_rand_chunks = n_rand_chunks
        self.p_ret_noise = p_ret_noise
        self.noise_std = noise_std
        self.percent_noise_std = percent_noise_std
        self.p_ret_blend = p_ret_blend
        self.blend_alpha = blend_alpha 
        self.n_blend = n_blend
        if self.sim_cutoff is not None or self.deduplicate:
            assert self.norm, "sim_cutoff requires norm=True"
        self.index_folder = Path(cache_path)
        self.index_path = self.index_folder / "knn.index"
        self.index_infos_path = self.index_folder / "index_infos.json"
        self.index_kwargs = index_kwargs if index_kwargs is not None else {}
        self.dedup_kwargs = dedup_kwargs if dedup_kwargs is not None else {}
        self.dataset_class = CacheDataset
        self.dataloader_class = DataLoader
        self.trjid_to_total_return = collections.defaultdict(int)
        self._is_ready = False
        if self.learnable_ret: 
            self.samples_class = TrajectoryReplayBufferSamplesWithContext
            self.dataset_class = CacheWithContextDataset

    @property
    def is_ready(self):
        return self._is_ready

    def setup_cache(self, keys, values, is_normed=False):
        """
        Sets up the cache with the given keys and values.

        Args:
            keys (numpy.ndarray): The keys to be used for building the index.
            values (dict): The values to be stored in the cache.

        Returns:
            None
        """
        self.task_to_value_idx = collections.defaultdict(list)
        self.trjid_to_value_idx = collections.defaultdict(list)
        # receive the keys and values from the Agent implementation
        print("Writing cache to: ", Path(self.index_folder).resolve())
        reset_folder_(self.index_folder)
        nb_cores = self.index_kwargs.get("nb_cores", None)
        self.index_kwargs["nb_cores"] = compute_nb_cores(nb_cores)
        if self.norm and not is_normed:
            keys = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
        build_index(
            keys,
            index_path=str(self.index_path),
            index_infos_path=str(self.index_infos_path),
            save_on_disk=True,
            **self.index_kwargs,
        )
        self.index = faiss_read_index(self.index_path, mmap=self.index_kwargs.get("should_be_memory_mappable", False))
        if self.use_gpu: 
            # if this fails, it means that the GPU version was not comp.
            # use_gpu can also be passed to build_index, but only has effect for IVF indices (for autofaiss reasons)
            assert (
                faiss.StandardGpuResources
            ), "FAISS was not compiled with GPU support, or loading _swigfaiss_gpu.so failed"
            # by default set to device 0 --> defined by CUDA_VISIBLE_DEVICES
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        self.index_type = self.index.__class__.__name__
        if self.nprobe is not None: 
            param_space = faiss.ParameterSpace if not self.use_gpu else faiss.GpuParameterSpace
            if "HNSW" in self.index_type:
                param_space().set_index_parameter(self.index, "efSearch", self.nprobe)
            elif "IVF" in self.index_type: 
                param_space().set_index_parameter(self.index, "nprobe", self.nprobe)
        
        # just keep values in memory for now
        self.values = values
        
        if self.exclude_same_trjs or self.deduplicate: 
            assert "trj_ids" in self.values.keys(), "No trj_ids were provided."
            for i, trj_id in enumerate(self.values["trj_ids"].numpy()):
                self.trjid_to_value_idx[trj_id].append(i)
        if "total_return" in self.values.keys(): 
            trj_ids = self.values["trj_ids"].numpy() if "trj_ids" in self.values.keys() else []
            for i, trj_id in enumerate(trj_ids):
                self.trjid_to_total_return[trj_id] = self.values["total_return"][i].item()
        if self.deduplicate: 
            assert self.index.metric_type == 0, "deduplication only makes sense for inner product."
            keys, values = self.deduplicate_cache(keys, **self.dedup_kwargs)
            self.deduplicate = False
            print("Re-setting up cache after deduplication.")
            self.setup_cache(keys, values, is_normed=True)
        if self.sample_kind == "task" or self.p_task_rand_trj > 0 or \
            self.p_sample_task_rand_trj > 0 or self.only_same_task or self.task_weight != 0 or \
            self.min_seq_len is not None:
            assert "task_ids" in self.values.keys(), "No task_ids were provided."
            for i, task_id in enumerate(self.values["task_ids"].numpy()):
                self.task_to_value_idx[task_id].append(i)
        if self.sample_kind == "return" or self.return_weight > 0 or self.min_return_for_ret is not None:
            assert "total_return" in self.values.keys(), "No total_return were provided."
        if self.sim_cutoff is not None: 
            assert self.index.metric_type == 0, "sim_cutoff only makes sense for inner product."
        if self.p_null_doc > 0 or self.force_improvement: 
            assert "attention_mask" in self.values.keys(), "No attention_mask was provided."
            for k, v in self.values.items(): 
                # attetion mask to 1s, otherwise exploding loss
                null_v = torch.zeros_like(v[0], device=v[0].device) if k != "attention_mask" \
                    else torch.ones_like(v[0], device=v[0].device)
                # null_v = torch.zeros_like(v[0], device=v[0].device)
                self.values[k] = torch.cat([v, null_v.unsqueeze(0)], dim=0)
            self.null_doc_idx = self.index.ntotal
        if self.min_seq_len or self.min_return_for_ret or self.sample_kind is not None or \
            self.p_task_rand_trj > 0 or self.p_sample_task_rand_trj > 0 or self.n_rand_chunks is not None or \
            self.p_ret_blend > 0:
            self.trj_id_sampler = TaskAwareIdSampler(
                self.task_to_value_idx, 
                total_returns=self.values["total_return"] if self.force_improvement else None
            )                 
        if self.p_ret_noise > 0: 
            # mean and std of stored keys 
            self.ret_noise_mean = np.mean(keys, axis=0)
            self.ret_noise_std = np.std(keys, axis=0)
        print("Cache is ready.")
        self._is_ready = True
        
    def make_dataset(self, env):
        """
        Creates a dataset from the stored trajectories and the given environment.
        Only change to parent method is that we pass cache_context_len to dataset.
        This requires refactoring.

        Args:
            env: The environment to use for creating the dataset.

        Returns:
            A dataset object that can be used for training a model on the stored trajectories.
        """
        if self.learnable_ret: 
            return self.dataset_class(
                self.trajectories, env, self.context_len, self.action_pad,
                trj_lengths=self.trajectory_lengths, trj_sample_kwargs=self.trj_sample_kwargs,
                max_state_dim=self.max_state_dim, max_act_dim=self.max_act_dim, to_rgb=self.to_rgb,
                transforms=self.transforms, task_to_trj=self.task_to_trj, trj_to_task=self.trj_to_task,
                seqs_per_sample=self.seqs_per_sample, seq_sample_kind=self.seq_sample_kind, 
                last_trj_mask=self.last_trj_mask, p_mask=self.p_mask, full_context_trjs=self.full_context_trjs,
                var_context_trjs=self.var_context_trjs, p_rand_trj=self.p_rand_trj,
                cache_context_len=self.cache_context_len, cache_steps=self.cache_steps,
                full_context_len=self.full_context_len, future_context_len=self.future_context_len,
                dynamic_context_len=self.dynamic_context_len, rand_first_chunk=self.rand_first_chunk,
            )
        super().make_dataset(env)

    def make_sampler(self, dataset, trajectory_probs, batch_size):
        """
        Creates a sampler for the given dataset, trajectory probabilities, and batch size.

        Args:
            dataset: The dataset to sample from.
            trajectory_probs: The probabilities of each trajectory.
            batch_size: The size of each batch.

        Returns:
            An iterator that generates indices for the given dataset.
        """
        # every trajectory is sampled as often as it's length
        indices = []
        for i, trj in enumerate(self.trajectories):
            trj_len = len(trj) if isinstance(trj, Trajectory) else self.trajectory_lengths[str(trj)] 
            trj_len //= self.cache_steps 
            indices += [i] * trj_len
            # indices += [i] * (trj_len + 1)
        return iter(indices)

    def query_cache(self, query, k, p_mask=0, reshape_context=False, compute_normed_distances=False,
                    only_same_task=False, task_id=None, trj_id=None, timesteps=None, total_returns=None,
                    idx_precalc=None, distances_precalc=None, trj_seed=None):
        """
        Retrieve the top-k items from the cache that are closest to the given query.

        Args:
            query (np.ndarray): The query to search for in the cache. Shape: [batch_size, query_dim].
            k (int): The number of items to retrieve from the cache.
            reshape_context (bool, optional): Whether to reshape the retrieved items to match the query shape. Defaults to False.
            compute_normed_distances (bool, optional): Whether to compute the normed distances between the query and the retrieved items. Defaults to False.
            same_task_only (bool, optional): Wheter to only retrieve from same task. Can be used during eval.
            task_id (int, optional): The IDs of the task of the queries.
            trj_id (int, optional): The IDs of the trajectories of the queries.

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]: A tuple containing:
                - distances (np.ndarray): The distances between the query and the retrieved items. Shape: [batch_size, k].
                - idx (np.ndarray): The indices of the retrieved items in the cache. Shape: [batch_size, k].
                - vals (Dict[str, np.ndarray]): A dictionary containing the retrieved items for each key in the cache. The shape of each item depends on the key.
                - normed_distances (np.ndarray, optional): The normed distances between the query and the retrieved items. Shape: [batch_size, k].
        """
        # all cases in which task_id/trj_id needs to be transferred
        if task_id is not None and (self.sample_kind == "task" or self.p_sample_task_rand_trj > 0 or \
            self.p_task_rand_trj > 0 or self.min_seq_len is not None or self.task_weight != 0 or \
            self.pos_weight != 0): 
            task_id = task_id.detach().cpu()
        if trj_id is not None and (self.exclude_same_trjs  or self.pos_weight != 0):
            trj_id = trj_id.detach().cpu()
        
        if idx_precalc is None and distances_precalc is None:
            if self.norm:
                query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
            if self.p_ret_blend > 0:
                if self.n_blend > 1:
                    assert self.p_ret_blend == 1 and self.reweight
                    query = np.repeat(query, self.n_blend, axis=0)
                    task_id = torch.repeat_interleave(task_id, self.n_blend, dim=0)
                    if total_returns is not None:
                        total_returns = torch.repeat_interleave(total_returns, self.n_blend, dim=0)
                    if trj_seed is not None: 
                        trj_seed = torch.repeat_interleave(trj_seed, self.n_blend, dim=0)
                    if trj_id is not None: 
                        trj_id = torch.repeat_interleave(trj_id, self.n_blend, dim=0)
                query = self.blend_with_rand_key(query, task_id)
                if self.norm: 
                    # norm blended query
                    query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
            if self.p_ret_noise > 0: 
                noise_idx = np.random.rand(query.shape[0]) < self.p_ret_noise
                n_noise = np.sum(noise_idx)
                if n_noise: 
                    if self.noise_std > 0:  
                        noise = np.random.normal(0, self.noise_std, size=(n_noise, query.shape[1]))
                    else: 
                        noise = np.random.normal(self.ret_noise_mean, self.ret_noise_std * self.percent_noise_std,
                                                 size=(n_noise, query.shape[1]))
                    query[noise_idx] += noise
                    if self.norm: 
                        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)

        idx, distances = self.retrieve_top_k(query, k, task_id, trj_id, 
                                             only_same_task=only_same_task, timesteps=timesteps,
                                             total_returns=total_returns,
                                             idx_precalc=idx_precalc, 
                                             distances_precalc=distances_precalc, 
                                             trj_seed=trj_seed)
        normed_distances = self.extract_normed_distances(query, idx, distances) if compute_normed_distances else None
        if reshape_context:
            vals = {}
            batch_size = idx.shape[0]
            for key, val in self.values.items():
                # self.values: [cache_size x seq_len x val_dim]
                # retrieved val: [batch_size, k, seq_len, val_dim] --> [batch_size, k * seq_len, val_dim]
                # timesteps and attention_masks are [batch_size, k, seq_len] --> treat separately
                val = val[idx.flatten()]
                if key == "states" and len(val.shape) > 4:
                    # treat images differently: [batch_size, k, seq_len, C, H, W] or [batch_size, k, C, H, W]
                    vals[key] = val.reshape(batch_size, -1, *val.shape[-3:])
                else:
                    vals[key] = (
                        val.reshape(batch_size, -1, val.shape[-1])
                        if len(val.shape) > 2
                        else val.reshape(batch_size, -1)
                    )
        else:
            vals = {k: val[idx] for k, val in self.values.items()}

        if "attention_mask" in vals and p_mask > 0: 
            attention_mask = vals["attention_mask"]
            mask = torch.bernoulli(torch.full(attention_mask.shape, p_mask))
            vals["attention_mask"] = attention_mask * (1 - mask)
            
        return distances, idx, vals, normed_distances

    def retrieve_top_k(self, query, k, task_id, trj_id, only_same_task=False,
                       timesteps=None, total_returns=None, idx_precalc=None, distances_precalc=None, 
                       trj_seed=None):
        """
        Retrieve the top k indices and distances from the cache.

        Args:
            query (numpy.ndarray): The query vector.
            k (int): The number of indices to retrieve.
            task_id (int): The IDs of the task corresponing to the queries.
            trj_id (int): The IDs of the trajectories corresponding to the queries.
            only_same_task (boolean): Whether to only retrieve from same task. Can be used during eval.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the retrieved indices and distances.
        """
        distances = None
        k = k if self.top_k is None else self.top_k
        if self.sample_kind == "task" or \
            (self.p_sample_task_rand_trj > 0 and np.random.rand() < self.p_sample_task_rand_trj):
            idx = self.sample_from_same_task(
                task_id, k if not self.reweight else self.reweight_top_k,
                total_returns=total_returns if self.force_improvement and total_returns is not None else None
            )
        elif self.sample_kind == "return":
            idx = self.sample_by_return(query, k)
        elif self.sample_kind == "random":
            idx = np.random.choice(self.index.ntotal, size=(query.shape[0], k))
        elif self.sample_kind == "max_return": 
            _, idx = torch.topk(self.values["total_return"], k=k)
        elif self.sample_kind == "last": 
            idx = np.arange(self.index.ntotal - k, self.index.ntotal)
        else:
            if idx_precalc is not None and distances_precalc is not None:
                # already precalculated, no need to retrieve again
                distances, idx = distances_precalc, idx_precalc
            elif self.only_same_task or only_same_task: 
                distances, idx = self.search_same_task_only(
                    query,
                    k if not self.sim_cutoff else self.sim_cutoff_k,
                    task_id,
                    trj_id
                )
            else: 
                params = None
                if (self.exclude_same_trjs and not self.only_same_task) and not self.use_gpu:                
                    # searchparameters have no effect on gpu indices for some reason 
                    # safeguard, to prevent model from retrieving from same trj index.
                    ids_to_exclude = self.compute_indices_to_exclude(trj_id)
                    ids_to_include = self.compute_indices_to_include() 
                    if len(ids_to_exclude) > 0: 
                        # for some reason SearchParameters have to be created inside same function as search() call 
                        selector = faiss.IDSelectorNot(faiss.IDSelectorArray(ids_to_exclude))
                        if len(ids_to_include) > 0:
                            selector = faiss.IDSelectorAnd(faiss.IDSelectorArray(ids_to_include), selector)
                        params = get_search_params_class(self.index, self.index_type, self.nprobe)(sel=selector)
                distances, idx = self.index.search(
                    query, 
                    k if not self.sim_cutoff else self.sim_cutoff_k,
                    params=params
                )
            if self.sim_cutoff is not None or (self.use_gpu and self.exclude_same_trjs and not self.reweight): 
                # similarity cutoff. in case of use_gpu, removes same trj indices if not reweight
                distances, idx = self.filter_by_similarity(
                    distances, idx, k, self.sim_cutoff, 
                    trj_id if self.exclude_same_trjs and self.use_gpu and not self.reweight else None
                )                    
            if self.reweight:
                assert k > 1, "k must be greater than 1, to weight by return."
                distances, idx = self.reweight_retrieved_trjs(distances, idx, k=self.reweight_top_k, task_id=task_id, 
                                                              trj_id=trj_id, trj_seed=trj_seed, timesteps=timesteps)
            
            distances, idx = distances.reshape(query.shape[0], -1), idx.reshape(query.shape[0], -1)
            if self.p_rand_trj > 0:
                # replace retrieved context trjs with random context trjs
                idx = self.replace_with_rand_indices(idx, idx.shape[1])
            if self.p_task_rand_trj > 0:
                # replace retrieved context trjs with random context trjs from same task
                idx = self.replace_with_rand_indices_same_task(
                    idx, idx.shape[1], task_id, total_returns=total_returns if self.force_improvement else None
                )
            if self.p_null_doc > 0: 
                idx = self.replace_with_null_doc(idx)
            if self.min_seq_len is not None and timesteps is not None: 
                idx = self.replace_with_rand_indices_same_task(idx, idx.shape[1], task_id, timesteps=timesteps)
            if self.min_return_for_ret is not None and total_returns is not None: 
                idx = self.replace_with_rand_indices_same_task(idx, idx.shape[1], task_id, total_returns=total_returns)
            if self.n_rand_chunks is not None: 
                idx = self.replace_with_rand_indices_same_task(idx, idx.shape[1], task_id, 
                                                               n_rand_chunks=self.n_rand_chunks)
        if self.p_rand_neighbour > 0: 
            idx = self.replace_with_rand_neighbour(idx)
        if self.n_blend > 1: 
            distances = distances.reshape(-1, distances.shape[-1] * self.n_blend)
            idx = idx.reshape(-1, idx.shape[-1] * self.n_blend)
            total_returns = total_returns[::self.n_blend]
        
        if idx.shape[1] > 1:
            if self.force_improvement and total_returns is not None: 
                # force stricly improvement trjs, otherwise padding trj
                is_higher = self.values["total_return"][idx.flatten()].reshape(idx.shape[0], -1).numpy() >= total_returns.unsqueeze(-1).cpu().numpy()
                idx[is_higher] = self.null_doc_idx
            # sort indices by return/timestep
            ret_return = self.values["total_return"][idx.flatten()].reshape(idx.shape[0], -1).numpy()
            ret_timestep = self.values["timesteps"][idx.flatten()][..., -1].reshape(idx.shape[0], -1).numpy()
            order = np.lexsort((ret_timestep, ret_return))
            idx = np.take_along_axis(idx, order, axis=1)
            distances = np.take_along_axis(distances, order, axis=1) if distances is not None else None
        return idx, distances

    def extract_normed_distances(self, query, idx, distances):
        if self.norm:
            # already normed, convert to [0, 1], as normed dot product is in range [-1, 1]
            normed_distances = (distances + 1) / 2
        else:
            # norm inner product to [0, 1] --> divide by the magnitudes of vectors
            idx = idx.flatten()
            query_norms = np.linalg.norm(query, axis=1, keepdims=True)
            key_norms = self.key_norms[idx].reshape(distances.shape[0], -1)
            normed_distances = distances / ((query_norms * key_norms) + 1e-8)
            normed_distances = (normed_distances + 1) / 2
        return normed_distances

    def sample_from_same_task(self, task_id, k, total_returns=None):
        assert task_id is not None, "Cache is set to sample from same task only, but no task_id was provided."
        idx = []
        task_id = task_id.numpy()
        total_returns = total_returns.detach().cpu().numpy() if total_returns is not None else None
        for i in range(task_id.shape[0]):
            idx.append(self.trj_id_sampler[task_id[i], total_returns[i] if total_returns is not None else None, k])            
        return np.stack(idx)

    def sample_by_return(self, query, k):
        total_return = self.values["total_return"] + 1e-8
        sample_weights = total_return / total_return.sum()
        sample_weights = torch.repeat_interleave(sample_weights.unsqueeze(0), query.shape[0], axis=0)
        return torch.multinomial(sample_weights, k, replacement=True)
    
    def replace_with_rand_indices(self, idx, k, n_rand_chunks=None):
        if n_rand_chunks is not None: 
            rand_mask = np.random.choice(idx.shape[1], size=(len(idx), n_rand_chunks), replace=False)
            rand_indices = np.random.choice(self.index.ntotal, size=(len(idx), n_rand_chunks), replace=True)
            idx[:, rand_mask] = rand_indices
        else: 
            rand_mask = np.random.random(len(idx)) < self.p_rand_trj
            rand_indices = np.random.choice(self.index.ntotal, size=(rand_mask.sum(), k), replace=True)
            idx[rand_mask] = rand_indices
        return idx

    def replace_with_rand_indices_same_task(self, idx, k, task_id, timesteps=None, 
                                            total_returns=None, n_rand_chunks=None):
        task_id = task_id.numpy()
        if self.force_improvement and total_returns is not None: 
            rand_mask = torch.rand(len(idx)) < self.p_task_rand_trj
        elif total_returns is not None: 
            rand_mask = total_returns < self.min_return_for_ret
        elif timesteps is not None: 
            rand_mask = timesteps[:, -1] < self.min_seq_len
        elif n_rand_chunks: 
            # rand chunks, per trj
            rand_mask = torch.ones(len(idx), dtype=bool)
        else: 
            rand_mask = torch.rand(len(idx)) < self.p_task_rand_trj
        for i in torch.where(rand_mask)[0]:
            if n_rand_chunks is not None:
                n_rand = n_rand_chunks
            else: 
                # n_rand = np.random.randint(1, k) if k > 1 else 1
                n_rand = k
            if self.force_improvement and total_returns is not None:
                val_idx = self.trj_id_sampler[task_id[i], total_returns[i], n_rand]
            else: 
                val_idx = self.trj_id_sampler[task_id[i], None, n_rand]
            if n_rand_chunks is not None:
                idx[i, np.random.choice(idx.shape[-1], n_rand_chunks, replace=False)] = val_idx
            else:    
                if len(val_idx) == idx.shape[-1]: 
                    idx[i] = val_idx
        return idx
    
    def replace_with_null_doc(self, idx):
        batch_size, k = idx
        idx = idx.flatten()
        rand_mask = np.random.random(len(idx)) < self.p_null_doc
        idx[rand_mask] = self.null_doc_idx
        return idx.reshape(batch_size, k)
    
    def replace_with_rand_neighbour(self, idx):
        batch_size, k = idx.shape
        idx = idx.flatten()
        rand_mask = np.random.random(len(idx)) < self.p_rand_neighbour
        rand_neighbour = np.random.choice(idx, size=rand_mask.sum())
        idx[rand_mask] = rand_neighbour
        return idx.reshape(batch_size, k)
    
    def blend_with_rand_key(self, query, task_id): 
        blend_idx = np.random.rand(query.shape[0]) < self.p_ret_blend
        for i in np.where(blend_idx)[0]:
            rand_idx = int(self.trj_id_sampler[task_id[i].item(), None, 1])
            query[i] = (1 - self.blend_alpha) * query[i] + self.blend_alpha * self.index.reconstruct(rand_idx)
        return query
    
    def filter_by_similarity(self, distances, idx, k, sim_cutoff=0.95, trj_id=None):
        """
        Filters the given distances and indices by similarity, keeping only the top-k most similar items.
        I.e., we retrieve more than k items, filter by similarity, and only take the top-k from the subselection.

        Assumes that the distances are already normed, and we are using ip metric.
        
        Args:
            distances (numpy.ndarray): An array of distances between the query and the items in the cache.
            idx (numpy.ndarray): An array of indices corresponding to the items in the cache.
            k (int): The number of items to keep.
            sim_cutoff (float, optional): The minimum similarity score required to keep an item. Defaults to 0.95.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the filtered distances and indices.
        """
        # downweight similarities that are > sim_cutoff
        distances[distances > sim_cutoff] = -1
        
        if trj_id is not None:
            # remove same trjs ids 
            trj_id = trj_id.numpy()
            for i, tid in enumerate(trj_id.flatten()):
                mask = np.isin(idx[i], self.trjid_to_value_idx[tid])
                idx[i][mask] = -1
                distances[i][mask] = -1

        # extract actual top k in numpy
        tmp = np.arange(distances.shape[0])[:, None]
        # need to do argsort, to have exact sort here
        if self.approx_sort: 
            topk_idx = np.argpartition(-distances, k, axis=1)[:, :k]
        else: 
            topk_idx = np.argsort(-distances, axis=1)[:, :k]
        distances = distances[tmp, topk_idx]
        idx = idx[tmp, topk_idx]
        return distances, idx

    def reweight_retrieved_trjs(self, distances, idx, k=1, task_id=None, trj_id=None, trj_seed=None, timesteps=None):
        """
        Reweights the retrieved trajectories. Distances, total returns and taskmatches are normalized to [0, 1]. 
        Final score is computed by:
            score = distance * distance weight + return * return weight + task * task weight.

        Args:
            distances: Computed distances to query.
            idx: Indices of retrieved trajectories.
            k (int, optional): Defaults to 1.

        """
        if len(idx.shape) == 1:
            idx = idx.reshape(1, -1)
        batch_dim = idx.shape[0]
        distances = torch.from_numpy(distances)
        
        # idx can be -1 in case of floating point errors, filter out
        invalid_idx_mask = (idx == -1).flatten()
        has_invalid_idx = invalid_idx_mask.any()
        
        # normalize distances and total_returns to [0, 1]
        if self.reweight_mult: 
            score = distances * self.dist_weight
        else: 
            score = self.min_max_norm_values(distances, inv=self.index.metric_type != 0, 
                                             mask=invalid_idx_mask if has_invalid_idx else None) * self.dist_weight
        
        if self.return_weight > 0:   
            # get total_returns, add small epsilon to preserve order when multipling by 0 return.
            total_returns = self.values["total_return"][idx.flatten()].reshape(batch_dim, -1) + 1e-8 
            if self.reweight_mult: 
                score *= (total_returns * self.return_weight)
            else: 
                return_score = self.min_max_norm_values(total_returns,
                                                        mask=invalid_idx_mask if has_invalid_idx else None)
                score += (return_score * self.return_weight)
        task_score = None
        if self.task_weight != 0: 
            ret_task_ids = self.values["task_ids"][idx.flatten()].reshape(batch_dim, -1)
            task_score = (ret_task_ids == task_id.unsqueeze(1)).float()
            if self.reweight_mult: 
                score *= (task_score * self.task_weight)
            else:
                score += (task_score * self.task_weight)
        if self.pos_weight > 0:
            # upweight trjs that come before the trj (AD-like)
            ret_trj_ids = self.values["trj_ids"][idx.flatten()].reshape(batch_dim, -1)
            # this should actually only count if it's also in the same task, otherwise doesn't make much sense
            if task_score is None: 
                ret_task_ids = self.values["task_ids"][idx.flatten()].reshape(batch_dim, -1)
                task_score = (ret_task_ids == task_id.unsqueeze(1)).float()
            pos_score = (ret_trj_ids < trj_id.unsqueeze(1)).float() * task_score
            if self.reweight_mult: 
                score *= (pos_score * self.pos_weight)
            else: 
                score += (pos_score * self.pos_weight)
        if self.improvement_weight > 0: 
            # upweight trjs with lower returns than input trj
            ret_total_returns = self.values["total_return"][idx.flatten()].reshape(batch_dim, -1)
            total_returns = self.extract_total_return_from_trj_ids(trj_id)
            improvement_score = ret_total_returns < total_returns.unsqueeze(1)
            if self.reweight_mult: 
                score *= (improvement_score * self.improvement_weight)
            else: 
                score += (improvement_score * self.improvement_weight)
        if self.seed_weight > 0: 
            ret_seeds = self.values["trj_seeds"][idx.flatten()].reshape(batch_dim, -1)
            seed_score = (ret_seeds == trj_seed.cpu().unsqueeze(1)).float()
            if self.reweight_mult:
                score *= (seed_score * self.seed_weight)
            else:
                score += (seed_score * self.seed_weight)

        if has_invalid_idx: 
            score = score.flatten()
            score[invalid_idx_mask] = 0
            score = score.reshape(batch_dim, -1)
        
        if self.exclude_same_trjs and self.use_gpu:
            # down-weight same trj ids - faiss-gpu does not support search parameters
            ret_trj_ids = self.values["trj_ids"][idx.flatten()].reshape(batch_dim, -1)
            mask = ret_trj_ids == trj_id.reshape(-1, 1)
            score[mask] = -1 if not self.reweight_mult else 0
            
        # extract trjs
        if self.reweight_sample: 
            # values need to be non-zero and non-negative
            topk_idx = torch.multinomial(score + 1e-8, k, replacement=True)
        else: 
            _, topk_idx = torch.topk(score, k=k, dim=1)
        topk_idx = topk_idx.numpy()
        score = np.take_along_axis(score, topk_idx, axis=1)
        idx = np.take_along_axis(idx, topk_idx, axis=1)
        return score, idx
    
    def deduplicate_cache(self, keys, k=10, sim_cutoff=0.98, batch_size=512,
                          same_task_only=True, lower_return_only=False): 
        remove_idx = set()
        for i in tqdm(range(0, self.index.ntotal, batch_size), desc="De-duplicating cache"): 
            trj_ids = [i for i in range(i, min(i + batch_size, self.index.ntotal)) if i not in remove_idx]
            if trj_ids == []: 
                continue  
            params = None
            # gpu does not support IDSelectors
            if not self.use_gpu:
                # safeguard, to prevent model from retrieving from same trj index.
                ids_to_exclude = self.compute_indices_to_exclude(self.values["trj_ids"][trj_ids])
                # do not consider remove indices
                ids_to_exclude += list(remove_idx)
                # for some reason SearchParameters have to be created inside same function as search() call 
                id_selector = faiss.IDSelectorNot(faiss.IDSelectorArray(ids_to_exclude))
                params = get_search_params_class(self.index, self.index_type, self.nprobe)(sel=id_selector)
                
            key = keys[trj_ids]
            distances, idx = self.index.search(key, k, params=params)
            if self.use_gpu: 
                # remove same trjs ids 
                trj_ids = np.array(trj_ids).reshape(-1, 1)
                mask = idx == trj_ids
                distances[mask] = -1
            if same_task_only and "task_ids" in self.values: 
                # remove only duplicates from same task --> set -1 for all others, don't remove
                task_ids, ret_task_ids = self.values["task_ids"][trj_ids], self.values["task_ids"][idx]
                mask = ret_task_ids != task_ids
                distances[mask] = -1
            if lower_return_only and "total_return" in self.values: 
                # only remove if return is lower or equal --> hence -1 only if retrieved return is higher
                # then it is prevented from removing. 
                total_returns, ret_total_returns = self.values["total_return"][trj_ids], self.values["total_return"][idx]
                mask = ret_total_returns > total_returns
                distances[mask] = -1
            mask = distances > sim_cutoff
            if mask.any():
                remove_idx.update(idx[mask])
        keep_idx = list(set(range(self.index.ntotal)) - remove_idx)
        print(f"Removed {len(remove_idx)} ({(round(len(remove_idx) / self.index.ntotal * 100, 2))}%) subsequences.")
        print(f"Keeping {len(keep_idx)} subsequences.")
        keys, values = keys[keep_idx], {k: v[keep_idx] for k, v in self.values.items()}
        return keys, values
    
    @staticmethod
    def min_max_norm_values(values, inv=False, mask=None):
        if inv: 
            values = 1 / (values + 1e-8)
        if mask is not None: 
            values_min, values_max = values.clone().flatten(), values.clone().flatten()
            values_min[mask] = float("inf")
            values_max[mask] = float("-inf")
            mins, _ = torch.min(values_min.reshape(values.shape[0], -1), dim=1, keepdim=True)
            maxs, _ = torch.max(values_max.reshape(values.shape[0], -1), dim=1, keepdim=True)
        else: 
            mins, _ = torch.min(values, dim=1, keepdim=True)
            maxs, _ = torch.max(values, dim=1, keepdim=True)
        values = (values - mins) / (maxs - mins + 1e-8)
        return values

    def compute_indices_to_exclude(self, trj_ids):
        trj_ids = trj_ids.numpy()
        ids_to_exclude = set()
        for tid in trj_ids.flatten():
            ids_to_exclude.update(self.trjid_to_value_idx[tid])
        return list(ids_to_exclude)

    def compute_indices_to_include(self):
        ids_to_include = []
        return ids_to_include
    
    def search_same_task_only(self, query, k, task_id, trj_id):
        distances, indices = [], []
        task_id = task_id.numpy()
        if self.exclude_same_trjs: 
            trj_id = trj_id.numpy()
        for i, (q, tid) in enumerate(zip(query, task_id)): 
            # for some reason SearchParameters have to be created inside same function as search() call
            selector = faiss.IDSelectorArray(self.task_to_value_idx[tid])
            if self.exclude_same_trjs:
                exclude_selector = faiss.IDSelectorNot(faiss.IDSelectorArray(self.trjid_to_value_idx[trj_id[i]]))
                selector = faiss.IDSelectorAnd(selector, exclude_selector)
            params = get_search_params_class(self.index, self.index_type, self.nprobe)(sel=selector)
            dist, idx = self.index.search(q.reshape(1, -1), k, params=params)
            distances.append(dist)
            indices.append(idx)
        return np.concatenate(distances), np.concatenate(indices)
            
    def add_to_cache(self, keys, values):
        # adds new keys/values to cache
        assert len(keys.shape) == 2, "Cache keys need to be of shape [batch_size, key_dim]."
        self.index.add(keys)
        for key, val in values.items():
            self.values[key] = torch.concat([self.values[key], val], axis=0)

    def should_store_trj(self, trj_return, reward_scale):
        if self.min_return is not None and trj_return < (self.min_return / reward_scale):
            print(f"Trajectory return {trj_return} is below min_return {self.min_return}.")
            return False
        return True

    def reset(self):
        print("Cleaning cache.")
        del self.index
        del self.values
        
    def extract_total_return_from_trj_ids(self, trj_ids):
        if len(self.trjid_to_value_idx.keys()) == 0: 
            return None
        return torch.tensor([self.trjid_to_total_return[tid.item()] for tid in trj_ids])
    
    def update_attributes(self, update_dict):
        self.task_weight = 0 
        self.pos_weight = 0 
        self.exclude_same_trjs = False
        self.min_seq_len = None
        self.sim_cutoff = None
        self.sample_kind = None
        for key, value in update_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Attribute {key} not found.")
        self.reweight = any([self.return_weight, self.task_weight, self.pos_weight])

    def cleanup_cache(self):
        print(f"Cleaning up cache folder: {Path(self.index_folder).resolve()}")
        reset_folder_(self.index_folder)


def get_search_params_class(index, index_type, nprobe=None):
    """
    Returns the search parameters class based on the index type.

    Args:
        index_type (str): The type of index.

    Returns:
        faiss.SearchParametersHNSW or faiss.SearchParametersIVF: The search parameters class.
    """
    if "HNSW" in index_type:
        efSearch = index.hnsw.efSearch if nprobe is None else nprobe
        param_class = functools.partial(faiss.SearchParametersHNSW, efSearch=efSearch)
    elif "IVF" in index_type:
        nprobe = index.nprobe if nprobe is None else nprobe
        param_class = functools.partial(faiss.SearchParametersIVF, nprobe=nprobe)
    else:
        param_class = faiss.SearchParameters
    return param_class

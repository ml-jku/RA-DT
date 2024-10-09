import collections
import torch
import numpy as np
from torch.utils.data import Sampler, DistributedSampler, Dataset
from typing import Iterator, Optional
from operator import itemgetter


class DomainWeightedRandomSampler(Sampler):

    def __init__(self, weights, num_samples, batch_size, replacement=True, generator=None):
        """
        Custom sampler class to sample batches from different domains alternatingly.
        This is necessary as different domains have different action and observation spaces. 
        For example, Atari has image observations and discrete actions, while Meta-World has 
        state-based in puts and continuous actions.
        To sample from differen domains, we would need to use a long sequence length and padding. 
        To avoid this, we sample from different domains alternatingly. After every batch, the sampler
        switches to the next domain

        Args:
            weights: Dict. Sampling weights for each domain.
            num_samples: Int. How many samples to draw from the dataset.
            batch_size: Int. How large the batches are. Required to switch domains after every batch.
            replacement: Bool. Whether to sample with replacement.
            generator: Random number generator.
            
        """
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.batch_size = batch_size
        self.num_domains = len(weights)
        self.num_samples_per_domain = self.num_samples // self.num_domains
        self.weights = {i: torch.as_tensor(w, dtype=torch.double) for i, w in weights.items()}
        self.domain_to_start_idx = collections.defaultdict(int)
        for i in range(1, self.num_domains):
            self.domain_to_start_idx[i] = self.domain_to_start_idx[i - 1] + len(self.weights[i - 1])
        
    def __iter__(self):
        # map domains idx to current sample idx within the domain
        domain_to_sample_idx = {i: 0 for i in self.weights.keys()}
        
        # generate indices per domain
        indices_per_domain = {}
        for domain, w in self.weights.items():
            indices_per_domain[domain] = torch.multinomial(
                w, self.num_samples_per_domain, self.replacement, generator=self.generator
            ) + self.domain_to_start_idx[domain]
    
        # yield the generated samples
        # after every batch, we switch to the next domain
        for i in range(len(self)):
            domain_idx = i // self.batch_size % self.num_domains
            sample_idx = domain_to_sample_idx[domain_idx]
            domain_to_sample_idx[domain_idx] += 1
            if domain_to_sample_idx[domain_idx] == self.num_samples_per_domain:
                domain_to_sample_idx[domain_idx] = 0
                print(f"Resetting domain_to_sample_idx: index {i}, domain_idx {domain_idx}, sample_idx: {sample_idx}")
            yield indices_per_domain[domain_idx][sample_idx]

    def __len__(self) -> int:
        return self.num_samples


class MixedBatchRandomSampler(Sampler):

    def __init__(self, weights, domain_weights, num_samples, batch_size, replacement=True, generator=None, ):
        """
        Custom sampler class to generated batches that contain samples from different domains in proportion to their
        prevelance in the training data.

        Args:
            weights: Dict. Sampling weights for each domain within that particular domain.
            domain_weights: Dict. Proportion for each domain in a single batch.
            num_samples: Int. How many samples to draw from the dataset.
            batch_size: Int. How large the batches are. Required to switch domains after every batch.
            replacement: Bool. Whether to sample with replacement.
            generator: Random number generator.
            
        """
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.batch_size = batch_size
        self.num_batches = self.num_samples // self.batch_size
        self.weights = {i: torch.as_tensor(w, dtype=torch.double) for i, w in weights.items()}
        self.domain_weights = domain_weights
        self.domain_to_start_idx = collections.defaultdict(int)
        for i in range(1, len(weights)):
            self.domain_to_start_idx[i] = self.domain_to_start_idx[i - 1] + len(self.weights[i - 1])
        
    def __iter__(self):
        # generate batch indices
        batch_indices = [[] for _ in range(self.num_batches * 2)]
        for domain, w in self.weights.items():
            # in proportion to domain_weights
            # create 2 number of samples per domain, to easily avoid OOB errors
            domain_weight = self.domain_weights[domain]
            samples_per_domain = max(round(self.num_samples * domain_weight), 1) * 2
            samples_per_batch = max(round(self.batch_size * domain_weight), 1)
            domain_indices = torch.multinomial(
                w, samples_per_domain, self.replacement, generator=self.generator
            ) + self.domain_to_start_idx[domain]
            for idx in range(self.num_batches * 2):
                indices = domain_indices[idx * samples_per_batch: (idx + 1) * samples_per_batch]
                batch_indices[idx].append(indices)

        # stack and flatten indices
        batch_indices = torch.cat([torch.cat(indices) for indices in batch_indices])
        
        for i in range(len(self)):
            yield batch_indices[i]

    def __len__(self) -> int:
        return self.num_samples
    
    
class DatasetFromSampler(Dataset):
    """
    
    Dataset to create indexes from `Sampler`.
    Copied from: https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/dataset.py
    
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            # the list() call blows up RAM considerably
            # for large datasets a better way is needed
            # self.sampler_list = list(self.sampler)
            sampler_list = [int(i) for i in self.sampler]
            self.sampler_list = torch.as_tensor(sampler_list)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)
    

class DistributedSamplerWrapper(DistributedSampler):
    """
    From: https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
    
    Unfortunately Pytorch DistributedSampler does not support weighted sampling or other kinds of sampling. 
    This issue is discussed at length here: 
        - https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/8
    An open feature request exists too: 
        - https://github.com/pytorch/pytorch/issues/23430
        - https://github.com/pytorch/pytorch/issues/77154
    
    
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class TaskAwareIdSampler: 

    def __init__(self, task_to_value_idx, n_samples_per_task=50000, n_quality_cat=10, total_returns=None): 
        """
        Sampler class for context trj ids. Pre-computes chunk indices at once, instead of 
        generating multiple times. 

        Args:
            task_to_value_idx: Dict. Contains task_idx-value_idx pairs. 
            n_samples_per_task: Int. Number of samples to pre-compute.
        """
        self.task_to_value_idx = task_to_value_idx
        self.n_samples_per_task = n_samples_per_task
        self.total_returns = total_returns
        self.n_quality_cat = n_quality_cat
        # contains either task id or (task_id, return) tuples
        self.task_to_samples = collections.defaultdict(list)
        self.task_to_idx_count = collections.defaultdict(int)
        # required when forcing improvement
        self.task_to_return_bounds = collections.defaultdict(list)
        self.generate_samples()
        
    def __getitem__(self, taskid_return_k):
        task_id, ret, k = taskid_return_k
        ret_cat = None
        if ret is not None: 
            # get upper bound for return category
            ret_cat = self.extract_return_category(task_id, ret)
            idx = (task_id, ret_cat)
        else: 
            idx = task_id
            if self.total_returns is not None: 
                idx = (task_id, np.random.randint(0, self.n_quality_cat))
        count = self.task_to_idx_count[idx]
        if count + k > self.n_samples_per_task:
            # re-generate
            self.generate_samples(task_id, category=ret_cat)
            self.task_to_idx_count[idx] = 0
            count = 0
        topk_idx = self.task_to_samples[idx][count: count + k]
        self.task_to_idx_count[idx] += k
        return topk_idx

    def __len__(self):
        pass

    def generate_samples(self, task_id=None, category=None): 
        print(f"Generating samples for task: {task_id if task_id is not None else 'all'} of category {category}.")
        task_id = [task_id] if task_id is not None else self.task_to_value_idx.keys()
        for tid in task_id: 
            task_indices = self.task_to_value_idx[tid]
            if self.total_returns is not None:
                # split up into total return categories - required in case for forcing improvement
                task_returns = self.total_returns[task_indices]
                if category is not None: 
                    upper = self.task_to_return_bounds[tid][category]
                    self.task_to_samples[(tid, category)] = np.random.choice(
                        np.array(task_indices)[task_returns <= upper].tolist(),
                        self.n_samples_per_task, replace=True
                    )
                else: 
                    percentiles = np.quantile(task_returns, np.arange(0, 1, 1 / self.n_quality_cat))
                    for i in range(self.n_quality_cat): 
                        # only generate samples where <= upper return bound. 
                        upper = percentiles[i]
                        self.task_to_samples[(tid, i)] = np.random.choice(
                            np.array(task_indices)[task_returns <= upper].tolist(),
                            self.n_samples_per_task, replace=True
                        )
                        self.task_to_return_bounds[tid].append(upper)
            else: 
                self.task_to_samples[tid] = np.random.choice(task_indices, 
                                                             self.n_samples_per_task, replace=True)

    def extract_return_category(self, task_id, ret): 
        task_rets = self.task_to_return_bounds[task_id]
        for i, upper in enumerate(task_rets): 
            if ret <= upper: 
                return i
        return len(task_rets) - 1

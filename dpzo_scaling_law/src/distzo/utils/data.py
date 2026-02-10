import torch
import math
from torch.utils.data import Sampler
import torch.distributed as dist

class PairDistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset.
    Crucially, it ensures that ranks within the same pair (e.g., 0 and 1)
    receive the SAME subset of data (indices), while different pairs receive
    different subsets.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if num_replicas % 2 != 0:
            raise ValueError(f"num_replicas ({num_replicas}) must be even for PairDistributedSampler")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        
        # Effective replicas = pairs
        self.num_pairs = self.num_replicas // 2
        self.pair_id = self.rank // 2
        
        # Calculate total size based on pairs, not individual ranks
        if self.drop_last and len(self.dataset) % self.num_pairs != 0:
            # Split samples closer to num_pairs
            self.num_samples = math.ceil(
                (len(self.dataset) - (len(self.dataset) % self.num_pairs)) / self.num_pairs
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_pairs)
            
        self.total_size = self.num_samples * self.num_pairs
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
            
        assert len(indices) == self.total_size

        # Subsample for this PAIR
        # The key difference from DistributedSampler: 
        # offset is based on self.pair_id, NOT self.rank
        indices = indices[self.pair_id : self.total_size : self.num_pairs]
        
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

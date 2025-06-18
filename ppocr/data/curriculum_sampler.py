from paddle.io import DistributedBatchSampler
from typing import Iterator
import math

class DistributedCurriculumLengthSampler(DistributedBatchSampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        super(DistributedCurriculumLengthSampler, self).__init__(
            dataset, batch_size, shuffle=shuffle, drop_last=drop_last
        )
        assert hasattr(self.dataset, "set_epoch")
        
    def __iter__(self) -> Iterator[list[int]]:
        self.set_epoch(self.epoch)
        return super().__iter__()

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

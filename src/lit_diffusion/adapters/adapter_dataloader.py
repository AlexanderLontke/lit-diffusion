from typing import Iterator, List

from torch.utils.data import DataLoader


class AdapterDataloader(Iterator):
    def __init__(self, original_dataloader: DataLoader, mapping: List[str], sort_mapping: bool = False):
        self.original_dataloader_it = original_dataloader.__iter__()
        self.mapping = sorted(mapping) if sort_mapping else mapping

    def __iter__(self):
        return self

    def __next__(self):
        next_batch = next(self.original_dataloader_it)
        result_batch = {}
        for i, name in enumerate(self.mapping):
            result_batch[name] = next_batch[i]
        return result_batch

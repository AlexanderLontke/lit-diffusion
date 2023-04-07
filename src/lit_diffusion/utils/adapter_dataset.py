from typing import Optional

from torch import nn
from torch.utils.data import Dataset


class AdapterDataset(Dataset):
    """
    Converts outputs from datasets which do not return dictionaries via the __getitem__ method into dictionaries
    so that they are accessible via a string key. This is not needed if the original dataset class' __getitem__
    method already returns a dictionary
    """

    def __init__(self, original_dataset: Dataset):
        # Instantiate dataset class
        self._dataset = original_dataset

    def __getitem__(self, item):
        dataset_output = self._dataset[item]
        print("Type:", type(dataset_output))
        # Handle single output case
        if not isinstance(dataset_output, list):
            dataset_output = [dataset_output]

        return {str(i): dataset_output[i] for i in range(len(dataset_output))}

    def __len__(self):
        return len(self._dataset)

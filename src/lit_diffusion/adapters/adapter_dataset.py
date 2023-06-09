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
        # Handle single output case
        if not isinstance(dataset_output, tuple):
            dataset_output = tuple(dataset_output)

        return {str(i): v for i, v in enumerate(dataset_output)}

    def __len__(self):
        return len(self._dataset)

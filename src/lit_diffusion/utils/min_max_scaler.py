from torch import nn


class MinMaxScaler(nn.Module):
    def __init__(
        self,
        minimum_value: float,
        maximum_value: float,
        interval_min: float = -1.0,
        interval_max: float = 1.0,
    ):
        super().__init__()
        # Store Scaler values
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.interval_min = interval_min
        self.interval_max = interval_max

    def forward(self, x):
        return ((x - self.minimum_value) / (self.maximum_value - self.minimum_value)) * (
            self.interval_max - self.interval_min
        ) + self.interval_min


if __name__ == '__main__':
    import torch

    mms = MinMaxScaler(minimum_value=0, maximum_value=255)
    sample = torch.rand(3, 120, 120,) * 255
    print(sample)
    scaled_sample = mms(sample)
    print("Min", scaled_sample.min(), "Max:", scaled_sample.max())
    print("Scaled sample", scaled_sample)

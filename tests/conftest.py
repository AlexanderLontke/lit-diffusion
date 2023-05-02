import pytest

import torch

EXAMPLE_BATCH_SIZE = 4
EXAMPLE_IMG_SHAPE = (3, 2, 2)

EXAMPLE_TIMESTEP_SCHEDULE_START = 0
EXAMPLE_TIMESTEP_SCHEDULE_END = 1000

T_KWARG_KEY = "t"
OTHER_IMG_KWARG_KEY = "other_img"


@pytest.fixture
def x_t_batch():
    return torch.randn(size=(EXAMPLE_BATCH_SIZE, *EXAMPLE_IMG_SHAPE))


@pytest.fixture
def t_batch():
    return torch.randint(
        low=EXAMPLE_TIMESTEP_SCHEDULE_START,
        high=EXAMPLE_TIMESTEP_SCHEDULE_END,
        size=(EXAMPLE_BATCH_SIZE,),
    )


@pytest.fixture
def batch_model_kwargs():
    return {
        OTHER_IMG_KWARG_KEY: torch.rand(
            size=(EXAMPLE_BATCH_SIZE, *EXAMPLE_IMG_SHAPE),
        )
    }

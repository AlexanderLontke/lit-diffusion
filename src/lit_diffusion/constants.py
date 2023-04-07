from enum import Enum

# Configuration keys for instantiation and training loop
# Seed
SEED_CONFIG_KEY = "seed"
# Data
DATASET_TRANSFORM_CONFIG_KEY = "dataset_transform"
TORCH_DATASET_CONFIG_KEY = "torch_dataset"
TORCH_DATA_LOADER_CONFIG_KEY = "torch_data_loader"
TORCH_DATASET_TRANSFORM_KEYWORD_CONFIG_KEY = (
    "transform_keyword"  # TODO: remove in favor of recursive instantiation
)

# Models
P_THETA_MODEL_CONFIG_KEY = "p_theta_model"
DIFFUSION_MODEL_CONFIG_KEY = "diffusion_model"
# Pytorch lightning
PL_TRAINER_CONFIG_KEY = "pl_trainer"
PL_WANDB_LOGGER_CONFIG_KEY = "pl_wandb_logger"
PL_MODEL_CHECKPOINT_CONFIG_KEY = "pl_checkpoint_callback"
# Sampling
SAMPLING_CONFIG_KEY = "sampling"
SAMPLING_SHAPE_CONFIG_KEY = "shape"
STRICT_CKPT_LOADING_CONFIG_KEY = "strict_ckpt_loading"
DEVICE_CONFIG_KEY = "device"
BATCH_SIZE_CONFIG_KEY = "batch_size"

# Fixed config keys for python module instantiation
STRING_PARAMS_CONFIG_KEY = "params"
PYTHON_CLASS_CONFIG_KEY = "module"


# Enum for DDPM target options
class DiffusionTarget(Enum):
    X_0 = "x_0"
    EPS = "eps"


# Metric keys
_TRAIN_PREFIX = "train/"
TRAINING_LOSS_METRIC_KEY = _TRAIN_PREFIX + "mse_loss"

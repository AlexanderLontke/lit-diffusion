from enum import Enum

# Configuration keys for instantiation and training loop
# Seed
SEED_CONFIG_KEY = "seed"
VERBOSE_INIT_CONFIG_KEY = "verbose_init"
# Data
TRAIN_TORCH_DATA_LOADER_CONFIG_KEY = "train_torch_data_loader"
VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY = "validation_torch_data_loader"

# Models
PL_MODULE_CONFIG_KEY = "pl_module"
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
SAFE_INTERMEDIARIES_CONFIG_KEY = "safe_intermediaries_every_n_steps"
CLIP_DENOISED_CONFIG_KEY = "clip_denoised"

# Fixed config keys for python module instantiation
PYTHON_CLASS_CONFIG_KEY = "module"
PYTHON_ARGS_CONFIG_KEY = "args"
PYTHON_KWARGS_CONFIG_KEY = "kwargs"
INSTANTIATE_DELAY_CONFIG_KEY = "delay"
CALL_FUNCTION_UPON_INSTANTIATION_KEY = "call"


# Enum for DDPM target options
class DiffusionTarget(Enum):
    X_0 = "x_0"
    EPS = "eps"


# Metric keys
LOGGING_TRAIN_PREFIX = "train/"
LOGGING_VAL_PREFIX = "val/"
TRAINING_LOSS_METRIC_KEY = "mse_loss"

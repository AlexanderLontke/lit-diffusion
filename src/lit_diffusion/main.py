import argparse, yaml
from pathlib import Path
from typing import Dict

# PyTorch imports
from torch.utils.data import DataLoader

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Util
from lit_diffusion.util import instantiate_python_class_from_string_config
from lit_diffusion.constants import (
    SEED_CONFIG_KEY,
    DATASET_TRANSFORM_CONFIG_KEY,
    TORCH_DATASET_CONFIG_KEY,
    TORCH_DATA_LOADER_CONFIG_KEY,
    P_THETA_MODEL_CONFIG_KEY,
    DIFFUSION_MODEL_CONFIG_KEY,
    PL_TRAINER_CONFIG_KEY,
    PL_WANDB_LOGGER_CONFIG_KEY,
    PL_MODEL_CHECKPOINT_CONFIG_KEY,
)

# Diffusion
from lit_diffusion.ddpm.ddpm import LitDDPM


def main(config: Dict):
    # Set seed
    pl.seed_everything(config[SEED_CONFIG_KEY])

    # Instantiate dataset Transform
    dataset_transform = instantiate_python_class_from_string_config(
        class_config=config[DATASET_TRANSFORM_CONFIG_KEY]
    )

    # Instantiate dataset
    train_dataset = instantiate_python_class_from_string_config(
        class_config=config[TORCH_DATASET_CONFIG_KEY], transform=dataset_transform
    )
    # Instantiate dataloader from data set
    train_data_loader = DataLoader(
        dataset=train_dataset, **config[TORCH_DATA_LOADER_CONFIG_KEY]
    )
    # Instantiate model approximating p_theta(x_{t-1}|x_t)
    p_theta_model = instantiate_python_class_from_string_config(
        class_config=config[P_THETA_MODEL_CONFIG_KEY]
    )
    # Instantiate DDPM class
    ddpm_pl_module = instantiate_python_class_from_string_config(
        class_config=config[DIFFUSION_MODEL_CONFIG_KEY],
        p_theta_model=p_theta_model,
    )

    # PL-Trainer with the following features:
    # - Model Summary
    # - Progressbar
    # - WAND logging
    # - Checkpointing
    trainer = pl.Trainer(
        **config[PL_TRAINER_CONFIG_KEY],
        logger=WandbLogger(**config[PL_WANDB_LOGGER_CONFIG_KEY], config=config),
        callbacks=[
            ModelCheckpoint(
                **config[PL_MODEL_CHECKPOINT_CONFIG_KEY],
            ),
        ],
    )
    # Run training
    trainer.fit(
        model=ddpm_pl_module,
        train_dataloaders=train_data_loader,
    )


if __name__ == "__main__":
    # Add run arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="config.yaml",
        help="Path to config.yaml",
        required=False,
    )

    # Parse run arguments
    args = parser.parse_args()

    # Load config file
    config_file_path = args.config
    with config_file_path.open("r") as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)

    # Run main function
    main(config)

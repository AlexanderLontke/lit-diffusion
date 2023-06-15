import argparse, yaml
from pathlib import Path
from typing import Dict

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Util
from lit_diffusion.util import instantiate_python_class_from_string_config
from lit_diffusion.constants import (
    SEED_CONFIG_KEY,
    VERBOSE_INIT_CONFIG_KEY,
    TRAIN_TORCH_DATA_LOADER_CONFIG_KEY,
    VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY,
    PL_MODULE_CONFIG_KEY,
    PL_TRAINER_CONFIG_KEY,
    PL_WANDB_LOGGER_CONFIG_KEY,
    PL_MODEL_CHECKPOINT_CONFIG_KEY,
    CUSTOM_CALLBACKS_CONFIG_KEY,
)


def main(config: Dict):
    # Set seed
    pl.seed_everything(config[SEED_CONFIG_KEY])
    verbose_init = config.get(VERBOSE_INIT_CONFIG_KEY, False)

    # Instantiate pytorch lightning module
    pl_module = instantiate_python_class_from_string_config(
        class_config=config[PL_MODULE_CONFIG_KEY],
        verbose=verbose_init,
    )

    # PL-Trainer with the following features:
    # - Progressbar (Default)
    # - Model Summary (Default)
    # - WAND logging
    # - Checkpointing
    # - Learning rate monitoring
    custom_callbacks = (
        [
            instantiate_python_class_from_string_config(
                class_config=class_config,
                verbose=verbose_init,
            )
            for class_config in config[CUSTOM_CALLBACKS_CONFIG_KEY]
        ]
        if CUSTOM_CALLBACKS_CONFIG_KEY in config.keys()
        else []
    )
    trainer = pl.Trainer(
        **config[PL_TRAINER_CONFIG_KEY],
        logger=WandbLogger(**config[PL_WANDB_LOGGER_CONFIG_KEY], config=config),
        callbacks=[
            ModelCheckpoint(
                **config[PL_MODEL_CHECKPOINT_CONFIG_KEY],
            ),
            LearningRateMonitor(),
            *custom_callbacks,
        ],
    )

    # Instantiate train dataloader
    train_dataloader = instantiate_python_class_from_string_config(
        class_config=config[TRAIN_TORCH_DATA_LOADER_CONFIG_KEY], verbose=verbose_init
    )
    # Instantiate validation dataloader
    val_dataloader = (
        instantiate_python_class_from_string_config(
            class_config=config[VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY],
            verbose=verbose_init,
        )
        if VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY in config.keys()
        else None
    )

    # Run training
    trainer.fit(
        model=pl_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
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

import os
import yaml
import argparse
import copy
from typing import Dict, Optional, List
from pathlib import Path

import pandas as pd

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from lit_diffusion.util import instantiate_python_class_from_string_config
from lit_diffusion.constants import (
    DEVICE_CONFIG_KEY,
    VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY,
    PL_MODULE_CONFIG_KEY,
    PL_WANDB_LOGGER_CONFIG_KEY,
    WANDB_PROJECT_KEY,
    WANDB_NAME_KEY,
    PYTHON_KWARGS_CONFIG_KEY,
    PL_TRAINER_CONFIG_KEY,
    LAST_CKPT_NAME,
    MONITOR_KEY,
    MONITOR_MODE_KEY,
    EPOCH_KEY,
)


def get_best_epoch_through_wandb(
    wandb_run,
    checkpoint_path: Path,
    keys_of_interest: List[str],
    monitor: str,
    monitor_mode: str,
) -> Path:
    single_run_complete_history = []
    for x in wandb_run.scan_history(keys=keys_of_interest, page_size=10000):
        single_run_complete_history.append(x)
    history_df = pd.DataFrame(single_run_complete_history)
    best_epoch = getattr(history_df[monitor], f"idx{monitor_mode}")()
    file_name = [
        file_name
        for file_name in os.listdir(checkpoint_path)
        if file_name.startswith(f"epoch={int(best_epoch)}")
    ][0]
    return checkpoint_path / file_name


def get_best_epoch_through_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_callback_states = torch.load(checkpoint_path / LAST_CKPT_NAME)[
        "callbacks"
    ]
    checkpoint_callback_keys = [
        k for k in checkpoint_callback_states.keys() if k.startswith("ModelCheckpoint")
    ]
    assert len(checkpoint_callback_keys) == 1
    checkpoint_callback_key = checkpoint_callback_keys[0]
    checkpoint_callback_state = checkpoint_callback_states[checkpoint_callback_key]
    return Path(checkpoint_callback_state["best_model_path"])


def get_best_checkpoints(
    wandb_project_name: str,
    wandb_run_name: str,
    complete_config: Dict,
    wandb_sub_project_name: str,
    through_callback: bool = False,
    checkpoints_root_path: Optional[str] = None,
) -> List[Path]:
    # This is only necessary because the mode of the checkpoint callback was misconfigured
    api = wandb.Api()
    run_filter = {
        "$and": [
            {"display_name": {"$eq": wandb_run_name}},
            {"state": {"$eq": "finished"}},
        ]
    }
    monitor = complete_config[MONITOR_KEY]
    monitor_mode = complete_config[MONITOR_MODE_KEY]
    runs = [
        run
        for run in api.runs(
            f"{wandb_project_name}/{wandb_sub_project_name}", filters=run_filter
        )
    ]
    keys_of_interest = [EPOCH_KEY, monitor]

    best_checkpoint_paths = []
    # Get all data
    for run in runs:
        checkpoint_path = Path(f"{wandb_sub_project_name}/{run.id}/checkpoints/")
        if checkpoints_root_path is not None:
            checkpoint_path = Path(checkpoints_root_path) / checkpoint_path
        if through_callback:
            checkpoint_path = get_best_epoch_through_checkpoint(
                checkpoint_path=checkpoint_path
            )
        else:
            checkpoint_path = get_best_epoch_through_wandb(
                wandb_run=run,
                checkpoint_path=checkpoint_path,
                keys_of_interest=keys_of_interest,
                monitor=monitor,
                monitor_mode=monitor_mode,
            )
        best_checkpoint_paths.append(checkpoint_path)
    return best_checkpoint_paths


def run_test(
    complete_config: Dict,
    through_callback: bool,
    test_beton_file: Optional[Path] = None,
    eval_suffix: str = "-eval",
):
    # Get WANDB values
    wandb_sub_project_name = complete_config[WANDB_PROJECT_KEY]
    wandb_project_name = complete_config[PL_WANDB_LOGGER_CONFIG_KEY][WANDB_PROJECT_KEY]
    wandb_run_name = complete_config[PL_WANDB_LOGGER_CONFIG_KEY][WANDB_NAME_KEY]

    # Fetch best checkpoints
    best_checkpoints = get_best_checkpoints(
        wandb_project_name=wandb_project_name,
        wandb_run_name=wandb_run_name,
        complete_config=complete_config,
        wandb_sub_project_name=wandb_sub_project_name,
        through_callback=through_callback,
    )

    # Update WANDB config for eval run
    complete_config[PL_WANDB_LOGGER_CONFIG_KEY][WANDB_PROJECT_KEY] = (
        wandb_project_name + eval_suffix
        if not wandb_project_name.endswith(eval_suffix)
        else wandb_project_name
    )
    complete_config[PL_WANDB_LOGGER_CONFIG_KEY][WANDB_NAME_KEY] = (
        wandb_run_name + eval_suffix
        if not wandb_run_name.endswith(eval_suffix)
        else wandb_run_name
    )
    # Save new config
    original_config = copy.deepcopy(complete_config)
    for checkpoint_path in best_checkpoints:
        print("Running eval for", checkpoint_path)
        complete_config = copy.deepcopy(original_config)
        wandb_logger = WandbLogger(
            **complete_config[PL_WANDB_LOGGER_CONFIG_KEY], config=complete_config
        )
        pl_module = instantiate_python_class_from_string_config(
            class_config=complete_config[PL_MODULE_CONFIG_KEY]
        )
        pl_module = pl_module.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=complete_config[DEVICE_CONFIG_KEY],
            downstream_model=pl_module.downstream_model,
            learning_rate=pl_module.learning_rate,
            loss=pl_module.loss,
            target_key=pl_module.target_key,
            validation_metrics=pl_module.validation_metrics,
        )
        # Create test dataloader config
        test_dataloader_config = copy.deepcopy(
            complete_config[VALIDATION_TORCH_DATA_LOADER_CONFIG_KEY]
        )
        # [Hack] to be removed
        if test_beton_file is not None:
            test_dataloader_config[PYTHON_KWARGS_CONFIG_KEY]["fname"] = test_beton_file

        test_dataloader = instantiate_python_class_from_string_config(
            class_config=test_dataloader_config
        )
        trainer = pl.Trainer(
            logger=wandb_logger,
            **complete_config[PL_TRAINER_CONFIG_KEY],
        )
        trainer.test(model=pl_module, dataloaders=test_dataloader)
        wandb.finish()


if __name__ == '__main__':
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

    parser.add_argument(
        "-p",
        "--project-name",
        type=Path,
        default="ssl-diffusion",
        help="wandb-project-name",
        required=False,
    )

    parser.add_argument(
        "--through-callback",
        type=str,
        default="True",
        help="determines how the best checkpoint is selected",
        required=False,
    )

    # [Hack] to be removed
    parser.add_argument(
        "-t",
        "--test-beton-file",
        default=None,
        type=str,
        help="Beton file containing test dataset in FFCV format",
        required=False,
    )

    # Parse run arguments
    args = parser.parse_args()

    # Load config file
    config_file_path = args.config
    with config_file_path.open("r") as config_file:
        config = yaml.safe_load(config_file)

    # Select best checkpoint method
    through_callback = args.through_callback.lower() in ["true"]

    # [Hack] to be removed
    test_beton_file = args.test_beton_file

    run_test(
        complete_config=config,
        through_callback=through_callback,
        test_beton_file=test_beton_file,
    )

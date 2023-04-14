import argparse

import torch
import yaml

from pathlib import Path

import lit_diffusion.ddpm.lit_ddpm
from lit_diffusion.util import instantiate_python_class_from_string_config
from lit_diffusion.constants import (
    PL_MODULE_CONFIG_KEY,
    VERBOSE_INIT_CONFIG_KEY,
    SAMPLING_CONFIG_KEY,
    SAMPLING_SHAPE_CONFIG_KEY,
    STRICT_CKPT_LOADING_CONFIG_KEY,
    DEVICE_CONFIG_KEY,
    BATCH_SIZE_CONFIG_KEY,
    SAFE_INTERMEDIARIES_CONFIG_KEY,
    CLIP_DENOISED_CONFIG_KEY,
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
    parser.add_argument(
        "-p",
        "--ckpt-path",
        type=Path,
        help="Path to model_checkpoint",
        required=True,
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
    verbose_init = config.get(VERBOSE_INIT_CONFIG_KEY, False)

    # Instantiate diffusion model class
    pl_module: lit_diffusion.ddpm.lit_ddpm.LitDDPM = (
        instantiate_python_class_from_string_config(
            class_config=config[PL_MODULE_CONFIG_KEY],
            verbose=verbose_init
        )
    )
    # Load Module checkpoint
    checkpoint_path = args.ckpt_path
    pl_module.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=config[SAMPLING_CONFIG_KEY][STRICT_CKPT_LOADING_CONFIG_KEY],
        p_theta_model=pl_module.p_theta_model,
    )
    # Load Module onto device
    pl_module.to(torch.device(config[SAMPLING_CONFIG_KEY][DEVICE_CONFIG_KEY]))
    pl_module.clip_denoised = config[SAMPLING_CONFIG_KEY][CLIP_DENOISED_CONFIG_KEY]

    # Sample from model
    sampled_image = pl_module.p_sample_loop(
        shape=config[SAMPLING_CONFIG_KEY][SAMPLING_SHAPE_CONFIG_KEY],
        batch_size=config[SAMPLING_CONFIG_KEY][BATCH_SIZE_CONFIG_KEY],
        safe_intermediaries_every_n_steps=config[SAMPLING_CONFIG_KEY][
            SAFE_INTERMEDIARIES_CONFIG_KEY
        ],
    )
    torch.save(
        sampled_image,
        f=f"./sampled_images_clipped_{config[SAMPLING_CONFIG_KEY][CLIP_DENOISED_CONFIG_KEY]}.pt",
    )

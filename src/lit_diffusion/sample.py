import argparse
import yaml

from pathlib import Path
from PIL import Image

import torchvision.transforms.functional as F
import pytorch_lightning as pl

from lit_diffusion.util import instantiate_python_class_from_string_config
from lit_diffusion.constants import DIFFUSION_MODEL_CONFIG_KEY


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

    # Get pytorch lightning module from config
    pl_module: pl.LightningModule = instantiate_python_class_from_string_config(
        class_config=config[DIFFUSION_MODEL_CONFIG_KEY],
    )
    # Load Module checkpoint
    checkpoint_path = args.ckpt_path
    pl_module.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Sample from model
    sampled_image: Image.Image = F.to_pil_image(pic=pl_module.sample())
    sampled_image.show()

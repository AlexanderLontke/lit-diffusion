from enum import Enum


# Enum for DDPM target options
class DDPMDiffusionTarget(Enum):
    X_0 = "x_0"
    EPS = "eps"


# Enum for DDPM loss options
class DDPMLossType(Enum):
    VLB = "vlb"
    MSE = "mse"

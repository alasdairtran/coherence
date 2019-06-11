import random

import numpy as np
import torch


def seed_environment(seed: int):
    """Seed environment for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

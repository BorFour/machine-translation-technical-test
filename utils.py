import logging
import random

import numpy as np
import torch
from langdetect import DetectorFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

log = logging.getLogger("ea-test")


def fix_seeds(seed: int = 53):
    """Fix the seeds of pseudo-random number generators so experiments are reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    DetectorFactory.seed = seed  # Needed to make language detection deterministic

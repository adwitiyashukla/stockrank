"""Single entry point for making a run reproducible."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:  # noqa: BLE001 - torch is an optional dependency
        pass

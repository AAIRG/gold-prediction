"""Seeding, device selection, config loading.

TODO (M0):
    - set_seed(seed: int) -> None
        * torch.manual_seed, torch.cuda.manual_seed_all, numpy, random
        * needed to reproduce runs; the paper reports mean and standard error
          over 20 reruns for CRPS_sum (Table 1) and 3 reruns for CRPS
          (Table 4), so seeding per run matters if you want to report spread
    - get_device(prefer_cuda: bool = True) -> torch.device
    - load_config(path: str) -> dict
        * YAML loader for configs/*.yaml
"""

import random
import torch
import numpy as np
import yaml

def set_seed(seed: int) -> None:
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception as e:
        raise Exception(f"common - set_seed - Execution error: {e}") from e

def get_device(prefer_cuda: bool = True) -> torch.device:
    try:
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception as e:
        raise Exception(f"common - get_device - Execution error: {e}") from e

def load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"common - load_config - Execution error: {e}") from e

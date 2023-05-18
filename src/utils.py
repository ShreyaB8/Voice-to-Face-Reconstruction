import gc
from datetime import datetime, timezone

import torch


def clear_memory():
    """Clear unused CPU or GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_tensor_device(apple_silicon: bool = True) -> torch.device:
    """A function to detect and return
    the most efficient device available on the current machine.

    CUDA is the most preferred device.
    If Apple Silicon is available, MPS would be selected.
    """

    if torch.cuda.is_available():
        return torch.device('cuda')

    if apple_silicon:
        try:
            if torch.backends.mps.is_available():
                return torch.device('mps')
        except AttributeError:
            ...

    return torch.device('cpu')


def current_utc_time() -> str:
    """Return current time in UTC timezone as a string.
    Useful for logging ML experiments.
    """
    dtn = datetime.now(timezone.utc)
    return '-'.join(list(map(str, [
        dtn.year, dtn.month, dtn.day, dtn.hour, dtn.minute, dtn.second
    ])))

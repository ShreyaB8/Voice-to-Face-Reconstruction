import json
import random
from abc import ABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch.random


@dataclass(frozen=True)
class BaseConfig(ABC):
    def to_dict(self):
        return asdict(self)

    def to_json(self, json_file: Path):
        with open(json_file, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json(cls, json_file: Path):
        with open(json_file, 'r') as f:
            return cls(**json.load(f))


@dataclass(frozen=True)
class TrainingConfig(BaseConfig):
    random_seed: int
    batch_size: int
    epochs: int
    learning_rate: float

    mlp_hidden_size: int | Sequence[int]
    mlp_hidder_layer_num: int
    mlp_dropout_probability: float | Sequence[float]
    
    continuation_target: str
    continuation_epoch: str
    strict_continuation: bool
    
    def set_random_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.random.manual_seed(self.random_seed)
        

@dataclass(frozen=True)
class BaselineTrainingConfig(TrainingConfig):
    mlp_output_size: int

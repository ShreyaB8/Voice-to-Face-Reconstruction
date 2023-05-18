from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


class Eigenface:
    eigenface_components: int
    weight: Tensor  # image_size * eigenface_components

    def __init__(
        self, eigenface_weight: Path,
        *, image_size: int = 128 * 128
    ):  
        if not eigenface_weight.exists() or not eigenface_weight.is_file():
            raise FileNotFoundError()
        if eigenface_weight.suffix not in ('.npy', '.pth'):
            raise ValueError('The input file must be a NumPy array or a PyTorch Tensor.')
        
        if eigenface_weight.suffix == '.npy':
            weight = torch.from_numpy(np.load(eigenface_weight)).to(torch.float32)
        else:
            weight = torch.load(eigenface_weight).to(torch.float32)
        
        if image_size == weight.size(0):
            eigenface_components = weight.size(1)
        elif image_size == weight.size(1):
            eigenface_components = weight.size(0)
            weight: Tensor = weight.transpose(0, 1)
        else:
            raise ValueError()
        weight = weight.requires_grad_(False)
        
        self.eigenface_components = eigenface_components
        self.weight = weight
        
        print(f'Initializing Eigenface converter with {eigenface_components = }.')
    
    def to(self, device: torch.device | str):
        self.weight = self.weight.to(device)
        return self

    def face_to_eigenface(self, face: Tensor):
        return face @ self.weight
    
    def eigenface_to_face(self, eigenface: Tensor):
        return eigenface @ self.weight.T
    
    
class DumbEigenface(nn.Module):
    linear: nn.Linear

    def __init__(
        self, input_size: int,
        *, image_size: int = 128 * 128
    ):  
        super().__init__()
        self.linear = nn.Linear(input_size, image_size, bias=False)
        
    def forward(self, x: Tensor):
        return self.linear(x)

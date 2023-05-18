"""
The face generator is attributed to Yandong Wen.
Please refer to this file for info:
https://github.com/cmu-mlsp/reconstructing_faces_from_voices/blob/master/network.py
"""
import os
from functools import partial
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.hub import download_url_to_file

from ..utils import get_tensor_device

GENERATOR_CHECKPOINT_LINK = 'https://github.com/cmu-mlsp/reconstructing_faces_from_voices/blob/master/pretrained_models/generator.pth?raw=true'


class Generator(nn.Module):
    def __init__(self, input_channel: int, channels: list[int], output_channel: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[4], output_channel, 1, 1, 0, bias=True),
        )

    def forward(self, x: Tensor):
        x = self.model(x)
        return x

    
forge_generator = partial(Generator, 64, [1024, 512, 256, 128, 64], 3)


def forge_generator_with_parameters(
    checkpoint_file: Path = Path('./checkpoints/generator.pth'),
    device: torch.device = get_tensor_device()
):
    if not checkpoint_file.exists():
        print(f'You are about to initialize the face generator with trained parameters, '
              f'but the provided checkpoint file path {checkpoint_file} does not exist.{os.linesep}'
              f'Downloading the model checkpoint file...')
        checkpoint_file.parent.mkdir(exist_ok=True, parents=True)
        download_url_to_file(GENERATOR_CHECKPOINT_LINK, checkpoint_file)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f'{checkpoint_file} does not exists.')
    if not checkpoint_file.is_file():
        raise IsADirectoryError(f'{checkpoint_file} is not a file.')

    generator = forge_generator().to(device)
    generator.load_state_dict(
        torch.load(checkpoint_file, map_location=device)
    )

    return generator

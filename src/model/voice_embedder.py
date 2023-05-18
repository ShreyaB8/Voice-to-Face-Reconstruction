"""
The VoxCeleb Embedder is attributed to Yandong Wen.
Please refer to this file for info:
https://github.com/cmu-mlsp/reconstructing_faces_from_voices/blob/master/network.py
"""

import os
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.hub import download_url_to_file
from torch.nn.functional import avg_pool1d

from ..utils import get_tensor_device

VOICE_EMBEDDER_CHECKPOINT_LINK = 'https://github.com/cmu-mlsp/reconstructing_faces_from_voices/blob/master/pretrained_models/voice_embedding.pth?raw=true'
DEFAULT_OUTPUT_FEATURE_NUM = 64


class VoiceEmbedNet(nn.Module):
    model: nn.Sequential

    def __init__(
        self, input_channel: int,
        output_channel: int,
        channels: Sequence[int]
    ):
        if len(channels) != 4:
            raise ValueError()

        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True),
        )

    def forward(self, x: Tensor):
        x = self.model(x)
        x = avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1, 1, 1)
        return x

forge_voice_embed_net = partial(VoiceEmbedNet, 64, 64, (256, 384, 576, 864))


def forge_voice_embedder_with_parameters(
    checkpoint_file: Path = Path('./checkpoints/voice_embedding.pth'),
    device: torch.device = get_tensor_device()
):
    if not checkpoint_file.exists():
        print(f'You are about to initialize the voice embedder with trained parameters, '
              f'but the provided checkpoint file path {checkpoint_file} does not exists.{os.linesep}'
              f'Downloading the model checkpoint file...')
        checkpoint_file.parent.mkdir(exist_ok=True, parents=True)
        download_url_to_file(VOICE_EMBEDDER_CHECKPOINT_LINK, checkpoint_file)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f'{checkpoint_file} does not exists.')
    if not checkpoint_file.is_file():
        raise IsADirectoryError(f'{checkpoint_file} is not a file.')

    voice_embed_net = forge_voice_embed_net().to(device)
    voice_embed_net.load_state_dict(
        torch.load(checkpoint_file, map_location=device)
    )

    return voice_embed_net

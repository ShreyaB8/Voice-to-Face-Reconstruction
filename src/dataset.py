import os
from glob import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from tqdm.auto import tqdm

from .model.eigenface import Eigenface


class VoiceToFaceDataset(Dataset):
    data: pd.DataFrame
    images: dict[str, Tensor]

    def __init__(
        self,
        voice_folder: Path = Path('datasets/voices'),
        image_folder: Path = Path('datasets/images'),
        metadata_file: Path = Path('datasets/metadata.csv'),
        eigenface_converter: Eigenface | None = None,
    ) -> None:
        super().__init__()
        
        if not voice_folder.exists() or not voice_folder.is_dir():
            raise FileNotFoundError('The voice dataset is not found.')
        if not image_folder.exists() or not image_folder.is_dir():
            raise FileNotFoundError('The image dataset is not found.')
        if not metadata_file.exists() or not metadata_file.is_file():
            raise FileNotFoundError('The metadata file is not found.')
        
        print(f'Creating dataset from {metadata_file}.')
        
        if eigenface_converter is not None:
            print('An Eigenface converter is provided. '
                  'All face images will be converted to Eigenface.')
        
        voice_entries = frozenset(os.listdir(voice_folder))
        image_entries = frozenset(os.listdir(image_folder))
        metadata = pd.read_csv(metadata_file)

        entries = []
        images: dict[str, Tensor] = {}
        for _, data in tqdm(metadata.iterrows(), total=len(metadata), desc='Loading dataset...'):
            voice_id: str = data['VoxCeleb1 ID']
            image_id: str = data['VGGFace1 ID']
            if voice_id not in voice_entries or image_id not in image_entries:
                continue

            gender: str = data['Gender']
            nationality: str = data['Nationality']                
    
            voice_paths = sorted(glob(f'{voice_folder / voice_id}/**/*.npy'))
            curr_entries = []
            for voice_file in voice_paths:
                curr_entries.append((voice_file, voice_id, image_id, gender, nationality))            
            entries.extend(curr_entries)

            image_paths = sorted(glob(f'{image_folder / image_id}/*.jpg'))
            image_data = torch.stack([
                read_image(image_file, ImageReadMode.GRAY)
                    .flatten().to(torch.float32)
                for image_file in image_paths
            ])
            if eigenface_converter is not None:
                image_data = eigenface_converter.face_to_eigenface(image_data)
            images[image_id] = image_data
        
        self.data = pd.DataFrame.from_records(
            entries, columns=[
                'voice_file', 'voice_id', 'image_id', 'gender', 'nationality'
            ]
        )
        self.images = images
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data = self.data.iloc[index].to_list()
        voice_file: str = data[0]
        voice_id: str = data[1]
        image_id: str = data[2]
        
        voice_data = torch.from_numpy(
            np.load(voice_file)[300:800, :]
        ).to(torch.float32).T
        image_data = self.images[image_id]
        
        return voice_data, voice_id, image_data, image_id

    @staticmethod
    def collate_fn(batch: Sequence[tuple[Tensor, str, Tensor, str]]):
        return (
            torch.stack([b[0] for b in batch]),
            [b[1] for b in batch],
            [b[2] for b in batch],
            [b[3] for b in batch],
        )
    
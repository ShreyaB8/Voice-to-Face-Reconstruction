from argparse import Namespace
from pathlib import Path
from pprint import pprint

import torch
from einops import rearrange
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torchsummaryX import summary
from tqdm.auto import tqdm

import wandb

from .config import BaselineTrainingConfig, TrainingConfig
from .dataset import VoiceToFaceDataset
from .model.eigenface import DumbEigenface, Eigenface
from .model.loss import batched_average_l2_loss
from .model.mlp import MLP
from .model.voice_embedder import (DEFAULT_OUTPUT_FEATURE_NUM,
                                   forge_voice_embedder_with_parameters)
from .utils import clear_memory, current_utc_time

__all__ = ['train', 'baseline_train']


def train(args: Namespace):
    debug: bool = args.debug
    eigenface_weight: Path = args.eigenface_weight
    image_folder: Path = args.image_folder
    voice_folder: Path = args.voice_folder
    train_metadata_file: Path = args.train_metadata_file
    valid_metadata_file: Path = args.valid_metadata_file
    wandb_entity: str = args.wandb_entity
    wandb_project_name: str = args.wandb_project_name
    checkpoint_dir: Path = args.checkpoint_dir
    
    config = TrainingConfig(
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,

        mlp_hidden_size=args.mlp_hidden_size,
        mlp_hidder_layer_num=args.mlp_hidden_layer_num,
        mlp_dropout_probability=args.mlp_dropout_probability,
        
        continuation_target=args.continuation_target,
        continuation_epoch=args.continuation_epoch,
        strict_continuation=args.strict_continuation,
    )

    config.set_random_seed()
    device = torch.device('cuda')
    
    job_name = current_utc_time()
    job_dir = checkpoint_dir / job_name
    
    if debug:
        print('Currently, the training routine is in debug mode. '
              'No checkpoints would be saved, '
              'and no experiment results would be logged to WAndB')
    
    voice_embedder = forge_voice_embedder_with_parameters().to(device).eval()
    eigenface_converter = Eigenface(eigenface_weight)
    mlp = MLP(
        DEFAULT_OUTPUT_FEATURE_NUM,
        eigenface_converter.eigenface_components, config.mlp_hidden_size,
        config.mlp_hidder_layer_num, config.mlp_dropout_probability
    ).to(device).train()
    
    if config.continuation_target and config.continuation_epoch:
        ckpt_file = checkpoint_dir / config.continuation_target / \
            f'mlp-{config.continuation_epoch}.pth'
        mlp.load_state_dict(
            torch.load(ckpt_file, map_location=device),
            strict=config.strict_continuation
        )
        print(f'Loaded parameters from {config.continuation_target} epoch {config.continuation_epoch}.')
    
    optimizer = AdamW(mlp.parameters(), config.learning_rate)
    
    training_dataset = VoiceToFaceDataset(voice_folder, image_folder, train_metadata_file, eigenface_converter)
    validating_dataset = VoiceToFaceDataset(voice_folder, image_folder, valid_metadata_file, eigenface_converter)
    dataloader_kwargs = dict(
        batch_size=config.batch_size, shuffle=True,
        pin_memory=True, num_workers=4
    )
    training_dataloader = DataLoader(training_dataset, collate_fn=training_dataset.collate_fn, **dataloader_kwargs)
    validating_dataloader = DataLoader(validating_dataset, collate_fn=validating_dataset.collate_fn, **dataloader_kwargs)

    for voice_data, _, _, _ in training_dataloader:
        with torch.no_grad():
            voice_embeddings = voice_embedder(voice_data.to(device))
            voice_embeddings = rearrange(voice_embeddings, 'N C 1 1 -> N C')
            summary_data_frame = summary(mlp, voice_embeddings)
        break
    else:
        raise RuntimeError()
    
    if not debug:
        job_dir.mkdir(exist_ok=True, parents=True)
        config.to_json(job_dir / 'config.json')
        summary_data_frame.to_json(job_dir / 'model-summary.json')
        
        wandb.init(
            name=job_name,
            project=wandb_project_name,
            entity=wandb_entity,
            config=config.to_dict(),
        )
        wandb.watch(mlp, log='all')
    else:
        pprint(config.to_dict())
        
    for epoch in range(config.epochs):
        total_training_loss = 0.0
        total_validating_loss = 0.0
        
        clear_memory()
        mlp.train()
        training_bar = tqdm(
            training_dataloader,
            desc=f'{epoch = }. Training...',
            total=len(training_dataloader),
        )
        for voice_data, _, reference_eigenfaces, _ in training_bar:
            optimizer.zero_grad()

            voice_data: Tensor = voice_data.to(device)
            reference_eigenfaces: list[Tensor] = [
                e.to(device) for e in reference_eigenfaces
            ]

            voice_embeddings = voice_embedder(voice_data)
            voice_embeddings = rearrange(voice_embeddings, 'N C 1 1 -> N C')
            eigenfaces = mlp(voice_embeddings)
            loss = batched_average_l2_loss(eigenfaces, reference_eigenfaces)

            total_training_loss += loss.item()
            loss.backward()
            optimizer.step()

        clear_memory()
        mlp.eval()
        validating_bar = tqdm(
            validating_dataloader,
            desc=f'{epoch = }. Validating...',
            total=len(validating_dataloader),
        )
        with torch.inference_mode():
            for voice_data, _, reference_eigenfaces, _ in validating_bar:
                optimizer.zero_grad()

                voice_data: Tensor = voice_data.to(device)
                reference_eigenfaces: list[Tensor] = [
                    e.to(device) for e in reference_eigenfaces
                ]

                voice_embeddings = voice_embedder(voice_data)
                voice_embeddings = rearrange(voice_embeddings, 'N C 1 1 -> N C')
                eigenfaces = mlp(voice_embeddings)
                loss = batched_average_l2_loss(eigenfaces, reference_eigenfaces)

                total_validating_loss += loss.item()

        training_loss = total_training_loss / len(training_dataloader)
        validating_loss = total_validating_loss / len(validating_dataloader)
        
        results = {
            'Training Loss': training_loss,
            'Validating Loss': validating_loss,
        }
        if not debug:
            wandb.log(results)
            torch.save(mlp.state_dict(), job_dir / f'mlp-{epoch}.pth')
        else:
            pprint(results)
    
    if not debug:
        wandb.finish()
        
        
def baseline_train(args: Namespace):
    debug: bool = args.debug
    image_folder: Path = args.image_folder
    voice_folder: Path = args.voice_folder
    train_metadata_file: Path = args.train_metadata_file
    valid_metadata_file: Path = args.valid_metadata_file
    wandb_entity: str = args.wandb_entity
    wandb_project_name: str = args.wandb_project_name
    checkpoint_dir: Path = args.checkpoint_dir
    
    config = BaselineTrainingConfig(
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,

        mlp_hidden_size=args.mlp_hidden_size,
        mlp_hidder_layer_num=args.mlp_hidden_layer_num,
        mlp_dropout_probability=args.mlp_dropout_probability,
        mlp_output_size=args.mlp_output_size,
        
        continuation_target=args.continuation_target,
        continuation_epoch=args.continuation_epoch,
        strict_continuation=args.strict_continuation,
    )

    config.set_random_seed()
    device = torch.device('cuda')
    
    job_name = current_utc_time()
    job_dir = checkpoint_dir / job_name
    
    if debug:
        print('Currently, the training routine is in debug mode. '
              'No checkpoints would be saved, '
              'and no experiment results would be logged to WAndB')
    
    voice_embedder = forge_voice_embedder_with_parameters().to(device).eval()
    dumb_eigenface = DumbEigenface(config.mlp_output_size).to(device).train()
    mlp = MLP(
        DEFAULT_OUTPUT_FEATURE_NUM,
        config.mlp_output_size, config.mlp_hidden_size,
        config.mlp_hidder_layer_num, config.mlp_dropout_probability
    ).to(device).train()
    
    if config.continuation_target and config.continuation_epoch:
        raise ValueError('So far, I have not implement continuation for baseline training.')
    
    optimizer = AdamW([
        {'params': mlp.parameters()},
        {'params': dumb_eigenface.parameters()},
    ], config.learning_rate)
    
    training_dataset = VoiceToFaceDataset(voice_folder, image_folder, train_metadata_file)
    validating_dataset = VoiceToFaceDataset(voice_folder, image_folder, valid_metadata_file)
    dataloader_kwargs = dict(
        batch_size=config.batch_size, shuffle=True,
        pin_memory=True, num_workers=4
    )
    training_dataloader = DataLoader(
        training_dataset, collate_fn=training_dataset.collate_fn, **dataloader_kwargs
    )
    validating_dataloader = DataLoader(
        validating_dataset, collate_fn=validating_dataset.collate_fn, **dataloader_kwargs
    )

    # for voice_data, _, _, _ in training_dataloader:
    #     with torch.no_grad():
    #         voice_embeddings = voice_embedder(voice_data.to(device))
    #         voice_embeddings = rearrange(voice_embeddings, 'N C 1 1 -> N C')
    #         mlp_summary_data_frame = summary(mlp, voice_embeddings)
    #     break
    # else:
    #     raise RuntimeError()
    
    if not debug:
        job_dir.mkdir(exist_ok=True, parents=True)
        config.to_json(job_dir / 'config.json')
        # mlp_summary_data_frame.to_json(job_dir / 'model-summary.json')
        
        wandb.init(
            name=job_name,
            project=wandb_project_name,
            entity=wandb_entity,
            config=config.to_dict(),
        )
        wandb.watch(mlp, log='all')
    else:
        pprint(config.to_dict())
        
    for epoch in range(config.epochs):
        total_training_loss = 0.0
        total_validating_loss = 0.0
        
        clear_memory()
        mlp.train()
        dumb_eigenface.train()
        training_bar = tqdm(
            training_dataloader,
            desc=f'{epoch = }. Training...',
            total=len(training_dataloader),
        )
        for voice_data, _, reference_faces, _ in training_bar:
            optimizer.zero_grad()

            voice_data: Tensor = voice_data.to(device)
            reference_faces: list[Tensor] = [
                e.to(device) for e in reference_faces
            ]

            voice_embeddings = voice_embedder(voice_data)
            voice_embeddings = rearrange(voice_embeddings, 'N C 1 1 -> N C')
            eigenfaces = mlp(voice_embeddings)
            faces = dumb_eigenface(eigenfaces)
            loss = batched_average_l2_loss(faces, reference_faces)

            total_training_loss += loss.item()
            loss.backward()
            optimizer.step()

        clear_memory()
        mlp.eval()
        validating_bar = tqdm(
            validating_dataloader,
            desc=f'{epoch = }. Validating...',
            total=len(validating_dataloader),
        )
        with torch.inference_mode():
            for voice_data, _, reference_faces, _ in validating_bar:
                optimizer.zero_grad()

                voice_data: Tensor = voice_data.to(device)
                reference_faces: list[Tensor] = [
                    e.to(device) for e in reference_faces
                ]

                voice_embeddings = voice_embedder(voice_data)
                voice_embeddings = rearrange(voice_embeddings, 'N C 1 1 -> N C')
                eigenfaces = mlp(voice_embeddings)
                faces = dumb_eigenface(eigenfaces)
                loss = batched_average_l2_loss(faces, reference_faces)

                total_validating_loss += loss.item()

        training_loss = total_training_loss / len(training_dataloader)
        validating_loss = total_validating_loss / len(validating_dataloader)
        
        results = {
            'Training Loss': training_loss,
            'Validating Loss': validating_loss,
        }
        if not debug:
            wandb.log(results)
            torch.save(mlp.state_dict(), job_dir / f'mlp-{epoch}.pth')
            torch.save(dumb_eigenface.state_dict(), job_dir / f'dump-eigenface-{epoch}.pth')
        else:
            pprint(results)
    
    if not debug:
        wandb.finish()

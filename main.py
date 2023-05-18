import sys
from argparse import ArgumentParser
from pathlib import Path

from torch.cuda import is_available as cuda_is_available

from src import baseline_train, train

if __name__ != '__main__':
    raise RuntimeError('This main script can only be invoked as a main file.')
if sys.version_info.major != 3 and sys.version_info.minor != 10:
    raise RuntimeError('This project requires Python 3.10.')
if not cuda_is_available():
    raise RuntimeError('So far I have not implemented non-CUDA platform support.')

parser = ArgumentParser('Project Main Entry')
sub_parser = parser.add_subparsers(dest='action')

train_parser = sub_parser.add_parser('train')

train_parser.add_argument(
    '-d', '--debug', action='store_true',
    help='Start the training routine in debug mode.'
         'In debug mode, no checkpoints or WANDB records would be logged or saved.',
)

train_parser.add_argument('--random-seed', type=int, default=1)
train_parser.add_argument('--epochs', type=int, default=50)
train_parser.add_argument('--batch-size', type=int, default=1024)
train_parser.add_argument('--learning-rate', type=float, default=1e-3)

train_parser.add_argument('--mlp-hidden-size', type=int, nargs='+', default=5120)
train_parser.add_argument('--mlp-hidden-layer-num', type=int, default=6)
train_parser.add_argument('--mlp-dropout-probability', type=float, nargs='+', default=0.0)

train_parser.add_argument('--continuation-target', type=str, default='')
train_parser.add_argument('--continuation-epoch', type=str, default='')
train_parser.add_argument('--strict-continuation', action='store_true')

train_parser.add_argument('--eigenface-weight', type=Path, default=Path('checkpoints/input-15k-pc-5k.npy'))
train_parser.add_argument('--image-folder', type=Path, default=Path('datasets/images'))
train_parser.add_argument('--voice-folder', type=Path, default=Path('datasets/voices'))
train_parser.add_argument('--train-metadata-file', type=Path, default=Path('datasets/metadata-train.csv'))
train_parser.add_argument('--valid-metadata-file', type=Path, default=Path('datasets/metadata-valid.csv'))
train_parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/training'))
train_parser.add_argument('--wandb-entity', type=str, default='idl-voice-to-face')
train_parser.add_argument('--wandb-project-name', type=str, default='training')

baseline_train_parser = sub_parser.add_parser('baseline-train')

baseline_train_parser.add_argument(
    '-d', '--debug', action='store_true',
    help='Start the training routine in debug mode.'
         'In debug mode, no checkpoints or WANDB records would be logged or saved.',
)

baseline_train_parser.add_argument('--random-seed', type=int, default=1)
baseline_train_parser.add_argument('--epochs', type=int, default=50)
baseline_train_parser.add_argument('--batch-size', type=int, default=256)
baseline_train_parser.add_argument('--learning-rate', type=float, default=1e-3)

baseline_train_parser.add_argument('--mlp-hidden-size', type=int, nargs='+', default=5120)
baseline_train_parser.add_argument('--mlp-hidden-layer-num', type=int, default=6)
baseline_train_parser.add_argument('--mlp-dropout-probability', type=float, nargs='+', default=0.0)
baseline_train_parser.add_argument('--mlp-output-size', type=int, default=5000)

baseline_train_parser.add_argument('--continuation-target', type=str, default='')
baseline_train_parser.add_argument('--continuation-epoch', type=str, default='')
baseline_train_parser.add_argument('--strict-continuation', action='store_true')

baseline_train_parser.add_argument('--image-folder', type=Path, default=Path('datasets/images'))
baseline_train_parser.add_argument('--voice-folder', type=Path, default=Path('datasets/voices'))
baseline_train_parser.add_argument('--train-metadata-file', type=Path, default=Path('datasets/metadata-train.csv'))
baseline_train_parser.add_argument('--valid-metadata-file', type=Path, default=Path('datasets/metadata-valid.csv'))
baseline_train_parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/baseline-training'))
baseline_train_parser.add_argument('--wandb-entity', type=str, default='idl-voice-to-face')
baseline_train_parser.add_argument('--wandb-project-name', type=str, default='baseline-training')

args = parser.parse_args()
if args.action == 'train':
    train(args)
elif args.action == 'baseline-train':
    baseline_train(args)

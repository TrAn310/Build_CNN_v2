"""
Entry point train.
"""

import argparse
from Src.training.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img', type=str, required=True)
    parser.add_argument('--train_lbl', type=str, required=True)
    parser.add_argument('--val_img', type=str, required=True)
    parser.add_argument('--val_lbl', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--width_mult', type=float, default=1.0)
    args = parser.parse_args()
    train(vars(args))


if __name__ == '__main__':
    main()
    
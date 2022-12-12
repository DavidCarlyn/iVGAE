import random

from argparse import ArgumentParser

import torch
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--hidden_channels", type=int, default=10)
    parser.add_argument("--out_channels", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_samples", type=int, default=8000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dset", type=str, choices=["asia", "earthquake"], default="asia")
    parser.add_argument("--i_vars", type=str, nargs="+", default=[])
    parser.add_argument("--i_probs", type=float, nargs="+", default=[])

    return parser.parse_args()
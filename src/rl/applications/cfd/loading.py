import itertools
import os
from json import JSONDecodeError

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.xutils.io import read_json
from src.xutils.logging import get_logger

logger = get_logger(__file__)


def try_read_json(file, delete=False):
    try:
        return read_json(file)
    except JSONDecodeError:
        logger.error(f"failed to load json file {file}. deleting it.")
        if delete:
            os.remove(file)
        return None


def load_samples_from_directory(directory, n=None):
    files = os.listdir(directory)
    if n is not None:
        files = files[:n]

    files = [try_read_json(os.path.join(directory, f), delete=False) for f in files]
    return [f for f in files if f is not None]


def split_dataset(x, y, train_frac):
    n_train = int(len(x) * train_frac)
    x_train, y_train = x[:n_train], y[:n_train][..., None]
    x_val, y_val = x[n_train:], y[n_train:][..., None]
    return x_train, y_train, x_val, y_val


def load_data_from_directories(directories, n=None, train_frac=None):
    files = list(itertools.chain(*[load_samples_from_directory(d) for d in directories]))
    np.random.shuffle(files)

    if n is not None:
        files = files[:n]

    data = pd.DataFrame(files)

    pressure_side = np.array(data["blc_coef_ps"].to_list())
    suction_side = np.array(data["blc_coef_ss"].to_list())
    x = np.concatenate([pressure_side, suction_side], axis=1)
    y = data["drag"].values

    logger.info(f"training data x-shape: {x.shape}, y-shape: {y.shape}")

    data = (x, y) if train_frac is None else split_dataset(x, y, train_frac)
    return data


def load_datasets(config, device):
    x_train, y_train, x_val, y_val = load_data_from_directories(config.DATA_SOURCE_DIRECTORIES, train_frac=0.8)
    x_train, y_train, x_val, y_val = [torch.Tensor(x).to(device) for x in [x_train, y_train, x_val, y_val]]
    train_dataset = DataLoader(TensorDataset(x_train, y_train), batch_size=config.TRAINING_REWARD_BATCH_SIZE, shuffle=True)
    val_dataset = DataLoader(TensorDataset(x_val, y_val), batch_size=config.TRAINING_REWARD_BATCH_SIZE, shuffle=False)
    return train_dataset, val_dataset

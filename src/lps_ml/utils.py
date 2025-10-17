"""
Utils Module

This module provides utility functions
"""
import os
import random
import datetime
import shutil
import hashlib
import json

import numpy as np

import torch


def get_available_device() -> torch.device:
    """
    Get the available device for computation.

    Returns:
        torch.device: The available device, either 'cuda' (GPU) or 'cpu'.
    """
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def print_available_device():
    """ Print the available device for computation. """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("No GPU available, using CPU.")

def set_seed():
    """ Set random seed for reproducibility. """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def backup_folder(base_dir, time_str_format = "%Y%m%d-%H%M%S"):
    """Method to backup all files in a folder in a timestamp based folder

    Args:
        base_dir (_type_): Directory to backup
        time_str_format (str, optional): Time string format for the folder.
            Defaults to "%Y%m%d-%H%M%S".
    """
    backup_dir = os.path.join(base_dir, datetime.datetime.now().strftime(time_str_format))
    os.makedirs(backup_dir)

    contents = os.listdir(base_dir)
    for item in contents:
        item_path = os.path.join(base_dir, item)

        if os.path.isdir(item_path):
            try:
                datetime.datetime.strptime(item, time_str_format)
                continue
            except ValueError:
                pass
        shutil.move(item_path, backup_dir)


class Hashable:
    """ Class for serializing and creating child hashables. """

    def __hash__(self):
        cfg = self.__get_hash_base__()
        cfg_str = json.dumps(cfg, sort_keys=True, default=str)
        return int(hashlib.sha256(cfg_str.encode()).hexdigest(), 16)

    def __get_hash_base__(self):
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": self._get_params()
        }

    def _get_params(self):
        return self._serialize(self.__dict__)

    def _serialize(self, obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._serialize(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, Hashable):
            return hash(obj)
        else:
            return str(obj)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and hash(self) == hash(other)

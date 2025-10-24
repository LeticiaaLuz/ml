"""Module to unify torch device access
"""
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

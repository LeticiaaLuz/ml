"""Basic __init__.py
Allows to import the __all__ by folder name.
"""
from .time_processors import SlidingWindowResampler

__all__ = [
    "SlidingWindowResampler"
]

"""Time processors
"""
import typing

import numpy as np

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_signal
import lps_ml.core as ml_core

def window_slider(data: np.ndarray, window_size: int, overlap_size: int) -> typing.List[np.ndarray]:
    """ Function to implement a sliding window for a given data. """

    step = int(window_size - overlap_size)

    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start:start + window_size])

    return windows

class SlidingWindowResampler(ml_core.AudioProcessor):
    """ Simple processor to resample and fragment the audio data. """

    def __init__(self,
                 fs_out: lps_qty.Frequency,
                 duration: lps_qty.Time,
                 overlap: lps_qty.Time):
        self.fs_out = fs_out
        self.duration = duration
        self.overlap = overlap

    def process(self, fs: lps_qty.Frequency, data: np.ndarray) -> typing.List[np.ndarray]:

        decimated_signal = lps_signal.decimate(data, fs/self.fs_out)

        window_size = int(self.duration * self.fs_out)
        overlap_size = int(self.overlap * self.fs_out)

        return window_slider(data = decimated_signal,
                             window_size = window_size,
                             overlap_size = overlap_size)


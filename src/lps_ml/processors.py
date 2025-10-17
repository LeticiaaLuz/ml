"""Processors Module

Define AudioFileProcessor interface and some processors
"""
import abc
import typing

import numpy as np

import lps_sp.signal as lps_signal
import lps_utils.quantities as lps_qty

import lps_ml.utils as lps_utils

class AudioFileProcessor(lps_utils.Hashable):
    """ Abstract class to process and audio file and get training window. """

    @abc.abstractmethod
    def process(self, fs: lps_qty.Frequency, data: np.array) -> typing.List[np.array]:
        """
        Process an audio into training windows.

        Args:
            fs (lps_qty.Frequency):  Sampling frequency of the input signal.
            data (np.array): Audio as a 1D tensor.

        Returns:
            windows (list of np.array):  A list of processed audio windows
        """


class WindowingResampler(AudioFileProcessor):
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
        step = int(window_size - overlap_size)

        windows = []
        for start in range(0, len(decimated_signal) - window_size + 1, step):
            windows.append(decimated_signal[start:start + window_size])

        return windows

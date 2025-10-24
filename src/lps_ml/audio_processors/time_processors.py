"""Time processors
"""
import typing

import numpy as np

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_signal
import lps_ml.core as ml_core

class Resampler(ml_core.AudioPipeline):
    """ AudioPipeline to change sample frequency. """

    def __init__(self,
                 fs_out: lps_qty.Frequency):
        super().__init__()
        self.fs_out = fs_out

    def process(self, fs: lps_qty.Frequency, data: np.array) \
            -> typing.Tuple[lps_qty.Frequency, np.array]:
        """
        Process an audio into processed audios.
        """

        decimated_signal = lps_signal.decimate(data, fs/self.fs_out)
        return self.fs_out, decimated_signal

class TimeProcessor(ml_core.AudioProcessor):
    """ Simple time processor for resampling, pipelined, and sliding windowing. """

    def __init__(self,
                 duration: lps_qty.Time,
                 overlap: lps_qty.Time,
                 fs_out: lps_qty.Frequency,
                 pipelines: typing.List[ml_core.AudioPipeline] = None):
        super().__init__()
        self.duration = duration
        self.overlap = overlap
        self.pipelines = pipelines or []

        if fs_out is not None:
            self.pipelines = [Resampler(fs_out=fs_out)] + self.pipelines

    def process(self, fs: lps_qty.Frequency, data: np.array) -> typing.List[np.array]:

        for pipeline in self.pipelines:
            fs, data = pipeline.process(fs=fs, data=data)

        window_size = int(self.duration * fs)
        overlap_size = int(self.overlap * fs)
        step = int(window_size - overlap_size)

        windows = []
        for start in range(0, len(data) - window_size + 1, step):
            windows.append(data[start:start + window_size])

        return windows


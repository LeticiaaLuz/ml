"""
Audio dataloader Module

This module provides functionality to read .wav files in a dataset.
"""
import os
import typing
import abc

import numpy as np
import scipy.io.wavfile as scipy_wav
import torch
import h5py
import json
import hashlib
import tqdm

import lps_utils.quantities as lps_qty

def _get_iara_id(filename:str) -> int:
    return int(filename.rsplit('-',maxsplit=1)[-1])

def _default_id(filename: str) -> int:
    return int(filename)


class Hashable:

    def __hash__(self):
        cfg = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": self._get_params()
        }
        cfg_str = json.dumps(cfg, sort_keys=True, default=str)
        return int(hashlib.sha256(cfg_str.encode()).hexdigest(), 16)

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

class AudioFileLoader(Hashable):
    """ Class to find and load audio files """

    def __init__(self,
                 data_base_dir: str,
                 extract_id: typing.Callable[[str], int] = _default_id):
        self.data_base_dir = data_base_dir
        self.extract_id = extract_id
        self.file_dict = {}
        self._find_raw_files()

    def _get_params(self):
        return {
            'data_base_dir': self.data_base_dir,
            'file_dict': len(self.file_dict)
        }

    def _find_raw_files(self) -> str:
        for root, _, files in os.walk(self.data_base_dir):
            for file in files:
                filename, extension = os.path.splitext(file)
                if extension == ".wav":
                    self.file_dict[self.extract_id(filename)] = os.path.join(root, file)

    def load(self, file_id: int) -> typing.Tuple[lps_qty.Frequency, np.array]:
        """
        Function to load data from file

        Args:
            file_id (int): The ID to search for.

        Returns:
            typing.Tuple[
                lps_qty.Frequency:        sample frequency
                np.array:     data
            ]:
        """

        if file_id not in self.file_dict:
            raise UnboundLocalError(f'file {file_id} not found in {self.data_base_dir}')

        fs, data = scipy_wav.read(self.file_dict[file_id])

        if data.ndim != 1:
            data = data[:,0]

        return lps_qty.Frequency.hz(fs), data


    @staticmethod
    def iara(data_base_dir: str) -> 'AudioFileLoader':
        """ Get AudioLoader for IARA dataset. """
        return AudioFileLoader(data_base_dir=data_base_dir, extract_id=_get_iara_id)

class AudioFileProcessor(Hashable):
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


class AudioDataLoader(Hashable):
    """ Class to process and manage file of processed data. """

    def __init__(self,
                 file_dir: str,
                 file_loader: AudioFileLoader,
                 file_processor: AudioFileProcessor,
                 file_ids: typing.List[int]):

        self.file_dir = file_dir
        self.file_loader = file_loader
        self.file_processor = file_processor
        self.file_ids = file_ids

        self.file_name = os.path.join(file_dir, self._get_file_name())

        if not os.path.exists(self.file_name):
            self._process_and_save()

        with h5py.File(self.file_name, "r") as f:
            self.processed_to_file = f["processed_to_file"][:]
            self.file_to_processed = {
                int(fid): f["file_to_processed"][str(fid)][:]
                    for fid in f["file_to_processed"].keys()
            }


    def _get_file_name(self) -> str:
        return f"{hash(self)}.h5"

    def _process_and_save(self):

        os.makedirs(self.file_dir, exist_ok=True)

        with h5py.File(self.file_name, "w") as f:
            audio_dset = None
            processed_to_file = None
            group = f.create_group("file_to_processed")
            current_idx = 0
            max_windows_alloc = None

            for fid in tqdm.tqdm(self.file_ids, desc="Processing Files", leave=False, ncols=120):

                fs, data = self.file_loader.load(fid)
                windows = self.file_processor.process(fs, data)

                n_windows = len(windows)
                window_shape = windows[0].shape

                if audio_dset is None:

                    max_windows_alloc = n_windows * 2 * len(self.file_ids)
                    audio_dset = f.create_dataset(
                        "audio",
                        shape=(max_windows_alloc, *window_shape),
                        maxshape=(None, *window_shape),
                        dtype="float32",
                        compression=None,
                        chunks=(1, *window_shape)
                    )
                    processed_to_file = f.create_dataset(
                        "processed_to_file",
                        shape=(max_windows_alloc,),
                        maxshape=(None,),
                        dtype="int32"
                    )

                if current_idx + n_windows > audio_dset.shape[0]:
                    new_size = max(current_idx + n_windows, audio_dset.shape[0] * 2)
                    audio_dset.resize((new_size, *window_shape))
                    processed_to_file.resize((new_size,))

                for j, window in enumerate(windows):
                    audio_dset[current_idx + j, ...] = window
                    processed_to_file[current_idx + j] = fid

                group.create_dataset(str(fid),
                                     data=list(range(current_idx, current_idx + n_windows)))

                current_idx += n_windows

            audio_dset.resize((current_idx, *window_shape))
            processed_to_file.resize((current_idx,))

    def __len__(self):
        return len(self.processed_to_file)

    def __getitem__(self, idx):
        return self.load(idx)

    def load(self, processed_ids: typing.Union[int, typing.List[int]]):
        """
        Loads one or more windows from processed_ids
        """
        if isinstance(processed_ids, int):
            processed_ids = [processed_ids]

        with h5py.File(self.file_name, "r") as f:
            data = f["audio"][processed_ids, ...]

        return torch.from_numpy(data)

    def map_file_ids_to_processed_ids(self, file_ids: typing.List[int]) -> typing.List[int]:
        """
        Returns all processed_ids associated with a list of file_ids
        """
        proc_ids = []
        for fid in file_ids:
            if fid in self.file_to_processed:
                proc_ids.extend(self.file_to_processed[fid])
        return proc_ids

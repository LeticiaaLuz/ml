"""
Audio dataloader Module

This module provides functionality to read .wav files in a dataset.
"""
import os
import json
import typing
import multiprocessing

import tqdm
import numpy as np
import pandas as pd

import scipy.io.wavfile as scipy_wav

import torch
import torch.utils.data as torch_data
import lightning

import lps_utils.quantities as lps_qty
import lps_ml.databases.cv as lps_cv
import lps_ml.utils as lps_utils
import lps_ml.processors as lps_proc

def _get_iara_id(filename:str) -> int:
    return int(filename.rsplit('-',maxsplit=1)[-1])

def _default_id(filename: str) -> int:
    return int(filename)


class AudioFileLoader(lps_utils.Hashable):
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

class ProcessedDataset(torch_data.Dataset):
    """ Simple dataset for processed data in .npy format """

    def __init__(self, dataframe: pd.DataFrame, processed_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.processed_dir = processed_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fragment_path = os.path.join(self.processed_dir, f"{row['id_fragment']}.npy")
        fragment = np.load(fragment_path)
        if self.transform:
            fragment = self.transform(fragment)
        return torch.from_numpy(fragment).float(), row['target']

class AudioDataModule(lightning.LightningDataModule, lps_utils.Hashable):
    """ Basic DataModule for process and load audio datasets. """

    def __init__(self,
                 file_loader: AudioFileLoader,
                 file_processor: lps_proc.AudioFileProcessor,
                 file_ids: typing.List[int],
                 targets: typing.List[int],
                 processed_dir: str,
                 batch_size: int = 32,
                 num_workers: int = None,
                 cv: lps_cv.CrossValidator = None,
                 transform=None):
        super().__init__()
        self.file_loader = file_loader
        self.file_processor = file_processor
        self.file_ids = file_ids
        self.targets = targets
        self.batch_size = batch_size
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() // 2)
        self.cv = cv or lps_cv.HoldOutCV()
        self.transform = transform

        self.dataframe = None
        self.folds = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.processed_dir = os.path.join(processed_dir, f"{hash(self)}")
        self.csv_file = os.path.join(self.processed_dir, "description.csv")

    def _get_params(self):
        return {
            "file_loader": self.file_loader.__get_hash_base__(),
            "file_processor": self.file_processor.__get_hash_base__(),
            "file_ids": self.file_ids,
            "targets": self.targets,
        }

    def prepare_data(self):

        os.makedirs(self.processed_dir, exist_ok=True)

        if os.path.exists(self.csv_file):
            print(f"[prepare_data] CSV ready: {self.csv_file}, skipping processing chain.")
            return

        records = []
        fragment_idx = 0

        for file_idx, file_id in enumerate(tqdm.tqdm(
                                    self.file_ids, desc="Processando arquivos", ncols=120)):
            fs, data = self.file_loader.load(file_id)
            fragments = self.file_processor.process(fs, data)

            for frag in fragments:
                frag_path = os.path.join(self.processed_dir, f"{fragment_idx}.npy")
                np.save(frag_path, frag)
                records.append({
                    "id_fragment": fragment_idx,
                    "file_id": file_id,
                    "target": self.targets[file_idx]
                })
                fragment_idx += 1

        df = pd.DataFrame(records)
        df.to_csv(self.csv_file, index=False)
        print(f"[prepare_data] Dataset processed with {len(df)} fragments at {self.csv_file}.")

        description = self.__get_hash_base__()
        desc_file = os.path.join(self.processed_dir, "description.json")
        with open(desc_file, "w", encoding="utf-8") as f:
            json.dump(description, f, indent=4, ensure_ascii=False)
        print(f"[prepare_data] Description saved to {desc_file}")

    def setup(self, stage=None):
        """
        Loads the metadata CSV and generates cross-validation folds.
        """
        self.dataframe = pd.read_csv(self.csv_file)

        self.folds = self.cv.apply(self.file_ids, self.targets)
        self.set_fold(0)

    def set_fold(self, fold_idx: int):
        """
        Sets the active fold for training/validation/test split.

        Args:
            fold_idx: Index of the fold to activate.
        """
        if self.folds is None:
            raise RuntimeError("Folds not initialized. Call setup() first.")

        if not 0 <= fold_idx < len(self.folds):
            raise ValueError(f"Invalid fold index {fold_idx} (max {len(self.folds)-1}).")

        fold_map = self.folds[fold_idx]

        df = self.dataframe.copy()
        df["group"] = df["file_id"].map(fold_map)

        self.train_df = df[df["group"] == lps_cv.FoldRole.TRAIN]
        self.val_df = df[df["group"] == lps_cv.FoldRole.VALIDATION]
        self.test_df = df[df["group"] == lps_cv.FoldRole.TEST]

    def train_dataloader(self):
        return torch_data.DataLoader(
                ProcessedDataset(self.train_df, self.processed_dir, self.transform),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

    def val_dataloader(self):
        return torch_data.DataLoader(
                ProcessedDataset(self.val_df, self.processed_dir, self.transform),
                batch_size=self.batch_size,
                num_workers=self.num_workers)

    def test_dataloader(self):
        return torch_data.DataLoader(
                ProcessedDataset(self.test_df, self.processed_dir, self.transform),
                batch_size=self.batch_size,
                num_workers=self.num_workers)

"""
4 classes Module
"""
import os

import pandas as pd

import lps_ml.databases.loader as lps_loader
import lps_ml.processors as lps_proc
import lps_ml.databases.cv as lps_cv

class FourClasses(lps_loader.AudioDataModule):
    """ DataModule for 4 classes dataset. """

    def __init__(self,
                 file_processor: lps_proc.AudioFileProcessor,
                 processed_dir: str = "/data/Processed_data/4classes",
                 data_dir: str = "/data/4classes",
                 batch_size: int = 32,
                 cv: lps_cv.CrossValidator = None,
                 num_workers: int = None):

        info_file = os.path.join(os.path.dirname(__file__),
                                 "dataset_info",
                                 "four_classes.csv")
        df = pd.read_csv(info_file)

        df["Target"], class_list = pd.factorize(df["Class"])

        loader = lps_loader.AudioFileLoader(data_base_dir=data_dir,
                                   extract_id=lambda name: int(name[-2:]))

        super().__init__(file_loader = loader,
                         file_processor = file_processor,
                         file_ids = df["ID"].to_list(),
                         targets = df["Target"].to_list(),
                         processed_dir = processed_dir,
                         batch_size = batch_size,
                         num_workers = num_workers,
                         cv = cv)

        self.df = df
        self.class_names = list(class_list)

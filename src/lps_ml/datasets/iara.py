"""
IARA DataCollections
"""
import typing
import enum
import os

import pandas as pd
import numpy as np

import lps_utils.quantities as lps_qty
import lps_ml.core as ml_core
import lps_ml.datasets.selection as ml_sel

class DataCollection(enum.Enum):
    """Enum representing the different audio record collections of IARA."""
    A = 0
    OS_NEAR_CPA_IN = 0
    B = 1
    OS_NEAR_CPA_OUT = 1
    C = 2
    OS_FAR_CPA_IN = 2
    D = 3
    OS_FAR_CPA_OUT = 3
    OS_CPA_IN = 4
    OS_CPA_OUT = 5
    OS_SHIP = 6

    E = 7
    OS_BG = 7
    OS = 8

    F = 9
    GLIDER_CPA_IN = 9
    G = 10
    GLIDER_CPA_OUT = 10
    GLIDER_SHIP = 11

    H = 12
    GLIDER_BG = 12
    GLIDER = 13

    COMPLETE = 14

    def __str__(self) -> str:
        """Return a string representation of the collection."""
        return str(self.name).rsplit(".", maxsplit=1)[-1]

    def _get_info_filename(self) -> str:
        """Return the internal filename for collection information."""
        return os.path.join(os.path.dirname(__file__),
                            "dataset_info",
                            "iara.csv")

    def get_selection_str(self) -> str:
        """Get string to filter the 'Dataset' column."""
        if self == DataCollection.OS_CPA_IN:
            return DataCollection.A.get_selection_str() + \
                "|" + DataCollection.C.get_selection_str()

        if self == DataCollection.OS_CPA_OUT:
            return DataCollection.B.get_selection_str() + \
                 "|" + DataCollection.D.get_selection_str()

        if self == DataCollection.OS_SHIP:
            return DataCollection.OS_CPA_IN.get_selection_str() + \
                "|" + DataCollection.OS_CPA_OUT.get_selection_str()

        if self == DataCollection.OS:
            return DataCollection.OS_SHIP.get_selection_str()+ \
                "|" + DataCollection.OS_BG.get_selection_str()

        if self == DataCollection.GLIDER_SHIP:
            return DataCollection.F.get_selection_str() + \
                "|" + DataCollection.G.get_selection_str()

        if self == DataCollection.GLIDER:
            return DataCollection.GLIDER_SHIP.get_selection_str() + \
                "|" + DataCollection.GLIDER_BG.get_selection_str()

        return str(self.name).rsplit(".", maxsplit=1)[-1]

    def get_prettier_str(self) -> str:
        """Get a prettier string representation of the collection."""
        labels = [
            "near cpa in",
            "near cpa out",
            "far cpa in",
            "far cpa out",
            "cpa in",
            "cpa out",
            "os ship",
            "os bg",
            "os",
            "glider cpa in",
            "glider cpa out",
            "glider ship",
        ]
        return labels[self.value]

    def to_df(self) -> pd.DataFrame:
        """Get information about the collection as a DataFrame.
        Returns:
            pd.DataFrame: A DataFrame containing detailed information about the collection.
        """
        df = pd.read_csv(self._get_info_filename(), na_values=[" - "])
        return df.loc[df['Dataset'].str.contains(self.get_selection_str())]

DC = DataCollection

class ShipLength(enum.Enum):
    """ Enum class to represent ship length as in original paper. """
    # https://www.mdpi.com/2072-4292/11/3/353
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    BACKGROUND = 3

    @staticmethod
    def classify(ship_length: typing.Union[float, int, str, 'lps_qty.Distance']) -> 'ShipLength':
        """ Function to convert length in meter to the enum. """

        if isinstance(ship_length, lps_qty.Distance):
            return ShipLength.classify(ship_length.get_m())

        if isinstance(ship_length, str):
            try:
                return ShipLength.classify(float(ship_length))
            except ValueError:
                return ShipLength.classify(np.nan)

        if np.isnan(ship_length):
            return ShipLength.BACKGROUND

        if ship_length < 50:
            return ShipLength.SMALL
        if ship_length < 100:
            return ShipLength.MEDIUM

        return ShipLength.LARGE


class IARA(ml_core.AudioDataModule):
    """ DataModule for IARA dataset. """

    @staticmethod
    def loader(data_base_dir: str) -> ml_core.AudioFileLoader:
        """ Get AudioFileLoader for IARA dataset. """
        return ml_core.AudioFileLoader(data_base_dir=data_base_dir,
                extract_id=lambda filename: int(filename.rsplit('-', maxsplit=1)[-1]))

    @staticmethod
    def length_selector(colunm_id : str = "Length") -> ml_sel.Selection:
        """ Get Selection for IARA dataset. """
        return ml_sel.Selection(
                target = ml_sel.CallbackTarget(
                        n_targets = 4,
                        function = lambda df: ShipLength.classify(df[colunm_id]).value))

    def __init__(self,
                 file_processor: ml_core.AudioProcessor,
                 data_collection: DC = DC.OS,
                 processed_dir: str = "/data/Processed_data/IARA",
                 data_dir: str = "/data/IARA",
                 batch_size: int = 32,
                 cv: ml_core.CrossValidator = None,
                 selection: ml_sel.Selection = None,
                 num_workers: int = None):

        df = data_collection.to_df()
        selection = selection or IARA.length_selector()

        super().__init__(file_loader = IARA.loader(data_base_dir=data_dir),
                         file_processor = file_processor,
                         description_df = selection.apply(df),
                         processed_dir = processed_dir,
                         batch_size = batch_size,
                         num_workers = num_workers,
                         cv = cv)

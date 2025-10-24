"""
Selection Module
"""
import typing
import abc

import pandas as pd

class Filter():
    """Abstract base class representing a filter to apply on a collection."""

    @abc.abstractmethod
    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the collection based on selected values present in a column.

        Args:
            input_df (pd.DataFrame): The input DataFrame to be filtered.

        Returns:
            pd.DataFrame: A filtered DataFrame.
        """

class LabelFilter():
    """Class representing a filter based on values present in a column on a collection."""

    def __init__(self,
                 column: str,
                 values: typing.List[str]):
        """
        Parameters:
        - column (str): Name of the column for selection.
        - values (List[str]): List of values for selection.
        """
        self.column = column
        self.values = values

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the collection based on selected values present in a column.

        Args:
            input_df (pd.DataFrame): The input DataFrame to be filtered.

        Returns:
            pd.DataFrame: A filtered DataFrame containing only the rows with values
                        present in the specified column.
        """
        return input_df.loc[input_df[self.column].isin(self.values)]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LabelFilter):
            return (self.column == other.column and
                    self.values == other.values)
        return False

class Target(Filter):
    """ Abstract class to convert element of dataframes to targets. """

    DEFAULT_TARGET_HEADER = 'Target'
    def __init__(self,
                 n_targets: int,
                 include_others: bool):
        self.n_targets = n_targets
        self.include_others = include_others

    def get_n_targets(self) -> int:
        """Return the number of targets."""
        return self.n_targets + (1 if self.include_others else 0)

    @abc.abstractmethod
    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the labelling in a dataframe.

        Args:
            input_df (pd.DataFrame): The input DataFrame to be classified.

        Returns:
            pd.DataFrame: A DataFrame with target columns
        """

    def grouped_column(self) -> str:
        """ Function to define label header to grouped data. """
        return self.DEFAULT_TARGET_HEADER

class LabelTarget(LabelFilter, Target):
    """Class representing a training target for a dataset."""

    def __init__(self,
                 column: str,
                 values: typing.List[str],
                 include_others: bool = False):
        """
        Parameters:
        - column (str): Name of the column for selection.
        - values (List[str]): List of values for selection.
        - include_others (bool): Indicates whether other values should be compiled as one
            and included or discarded. Default is to discard.
        """
        LabelFilter.__init__(self, column=column, values=values)
        Target.__init__(self, n_targets=len(values), include_others=include_others)

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the labelling in a dataframe.

        Args:
            input_df (pd.DataFrame): The input DataFrame to be classified.

        Returns:
            pd.DataFrame: A DataFrame with target columns

        Notes:
            - The target values are assigned based on the unique values present in the 'self.column'
                of the DataFrame.
            - Each unique value is mapped to an integer index, starting from 0.
            - If a value is not found in the 'self.values' list, it is assigned the index equal
                to the length of 'self.values'.
        """
        if not self.include_others:
            input_df = LabelFilter.apply(self, input_df)

        input_df = input_df.assign(**{self.DEFAULT_TARGET_HEADER:
            input_df[self.column].map({value: index for index, value in enumerate(self.values)})})
        input_df[self.DEFAULT_TARGET_HEADER] = \
            input_df[self.DEFAULT_TARGET_HEADER].fillna(len(self.values))
        input_df[self.DEFAULT_TARGET_HEADER] = \
            input_df[self.DEFAULT_TARGET_HEADER].astype(int)
        return input_df

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Target):
            return (self.column == other.column and
                    self.values == other.values and
                    self.include_others == other.include_others)
        return False

    @classmethod
    def from_dataframe(cls,
                       input_df: pd.DataFrame,
                       column: str) -> 'LabelTarget':
        """
        Create a LabelTarget automatically from unique values in a column.
        """
        _, uniques = pd.factorize(input_df[column])
        return cls(column=column, values=uniques.tolist(), include_others=False)

class CallbackTarget(Target):
    """ Class to implement target based on a callback function. """

    def __init__(self,
                 n_targets : int,
                 function: typing.Callable[[pd.Series],int],
                 include_others: bool = False):
        super().__init__(n_targets=n_targets, include_others=include_others)
        self.function = function

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:

        input_df[self.DEFAULT_TARGET_HEADER] = input_df.apply(self.function, axis=1)

        if not self.include_others:
            input_df[self.DEFAULT_TARGET_HEADER] = \
                input_df[self.DEFAULT_TARGET_HEADER].fillna(self.n_targets)
        else:
            input_df = input_df.dropna(subset=[self.DEFAULT_TARGET_HEADER])

        input_df[self.DEFAULT_TARGET_HEADER] = \
            input_df[self.DEFAULT_TARGET_HEADER].astype(int)
        return input_df

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CallbackTarget):
            return (self.n_targets == other.n_targets and
                    self.include_others == other.include_others)
        return False

class Selection:
    """ Class representing a selection of data and define targets."""

    def __init__(self,
                 target: Target,
                 filters: typing.Union[typing.List[Filter], Filter] = None):

        self.target = target
        self.filters = [] if filters is None else \
                (filters if isinstance(filters, list) else [filters])

    def apply (self, input_df : pd.DataFrame) -> pd.DataFrame:
        """ Apply selection to an input_df. """

        df = input_df
        for filt in self.filters:
            df = filt.apply(df)

        return self.target.apply(df)

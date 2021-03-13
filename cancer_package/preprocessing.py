import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List


class BasicPreprocessing():
    """
    Preforms basic preprocessing steps on the patient data
    """
    def __init__(self, data: pd.DataFrame, nan_value: int) -> None:
        self.nan_value = nan_value
        self.data = self._rm_empty_cols( data)
        self.non_proteins = ['patient_id', 'category']
        self._identify_proteins()
        self._reorder_columns()
  
    @staticmethod
    def _rm_empty_cols(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes empty columns from df or where all values in the column 
        are identical.
        :param df: a DataFrame
        """
        n_distinct = df.nunique()
        return df.drop( n_distinct[ n_distinct <= 1].index, axis=1)

    def _identify_proteins(self) -> list:
        """
        Identifies which columns contein proteins.
        Populates self.proteins
        """
        is_protein = ~self.data.columns.isin(self.non_proteins)
        self.proteins = list(self.data.columns[is_protein])
  
    def _reorder_columns(self) -> None:
        """
        Puts non protein information before the protein's expression levels.
        Reorders self.data
        """
        self.data = self.data[self.non_proteins + self.proteins]

    def replace_nans(self) -> None:
        """
        Replace the numerical encoding of missing data with an actual NaN
        """
        self.data.replace(self.nan_value, np.nan, inplace=True)
  
    def rm_energy_proteins(self, enery_proteins: List[str]) -> None:
        """
        Remove the energy proteins from the data
        :param energy_proteins: list of energy proteins
        """
        self.data.drop(enery_proteins, axis=1, inplace=True, errors="ignore")
        self._identify_proteins()
  
    def _perc_not_nan_calc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the share of missing values.
        """
        return (~df[self.proteins].isin([self.nan_value, np.nan])).sum() / len(df)
  
    def rm_execess_nans(
        self,
        non_nan_threshold: float,
        by_group: bool = True) -> None:
        """
        Only keeps the proteins should that one of the categories has at least
        non_nan_threshold fraction of non NaN values
        e.g. consider a non_nan_threshold of 0.7. If there are 2 equal sized  
             categories A with 50% non-NaNs and B with 80%, hence the population 
             percentage of non-NaNs is 65%. Still the protein will be kept because 
             one of the categories has more than the 70% threshold.
        :param non_nan_threshold: minimum share of non-NaN values in at least 
            one category [0, 1)
        """
        assert 0 <= non_nan_threshold < 1, "non_nan_threshold must be in [0, 1)"
        if by_group:
            non_nan_by_category = (
                self.data
                .groupby("category")
                .apply(self._perc_not_nan_calc)
            )
            bellow_thresh = non_nan_by_category.max() <= non_nan_threshold
        else:
            non_nan_by_category = self._perc_not_nan_calc(self.data)
            bellow_thresh = non_nan_by_category <= non_nan_threshold
        self.data.drop(
            bellow_thresh[bellow_thresh].index,
            axis=1,
            inplace=True
        )
        self._identify_proteins()

    def non_nan_share_hist(self, bins: int = 10) -> None:
        """
        Plots a histogram with the share of non-NaN in the category with the least NaN
        """
        (
            self.data
            .groupby("category")
            .apply(self._perc_not_nan_calc)
            .max()
            .plot.hist(bins=bins)
        )
        plt.show()

    def organise_proteins(self, protein_group: pd.DataFrame) -> None:
        """
        Sort the columns in such way that the proteins from the same group are together.
        The sorting is done according to the DataFrame protein_group.
        :param protein_group: a df containing a column called Protein
        """
        self.proteins = list(
            protein_group.Protein[
                protein_group.Protein.isin(self.proteins)
            ]
        )
        self._reorder_columns()

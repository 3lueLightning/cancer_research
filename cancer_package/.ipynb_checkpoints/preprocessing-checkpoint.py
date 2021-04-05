import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List

from . import constants
from . import utilities


class BasicPreprocessing():
    """
    Preforms basic preprocessing steps on the patient data
    """
    def __init__(
            self,
            data: pd.DataFrame, 
            nan_value: int,
            use_energy_proteins: bool = True) -> None:
        n_cols = data.shape[1]
        self.data = self._rm_empty_cols( data)
        print(f"rm {n_cols - self.data.shape[1]} proteins with no data")
        self.proteins = [] 
        self.non_proteins = ['patient_id', 'category']
        self.nan_value = nan_value
        self.use_energy_proteins = use_energy_proteins
        # empty variables
        self.formulas_df = pd.DataFrame()
        self.equation_mapping = {}
        self.clean_equation_mapping = {}
        # processing
        self._identify_proteins()
        if not self.use_energy_proteins:
            n_cols = self.data.shape[1]
            self.rm_energy_proteins()
            print(f"rm {n_cols - self.data.shape[1]} energy proteins")
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
        if self.proteins:
            still_present = pd.Series(self.proteins).isin(self.data.columns)
            self.proteins = list(pd.Series(self.proteins)[still_present])
        else:
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
    
    
    def rm_energy_proteins(self) -> None:
        """
        Remove the energy proteins from the data
        """
        self.data.drop(constants.ENERGY_PROTEINS, axis=1, inplace=True, errors="ignore")
        self._identify_proteins()
    
    
    def _perc_not_nan_calc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the share of missing values.
        """
        return (~df[self.proteins].isin([self.nan_value, np.nan])).sum() / len(df)
    
    
    def rm_execess_nans(
        self,
        non_nan_threshold: float,
        by_group: bool = True,
        redefine_proteins: bool = True) -> None:
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
        proteins_in_group = protein_group.Protein.isin(self.proteins)
        group_proteins = list(protein_group.Protein[proteins_in_group])
        proteins_not_in_group = ~pd.Series(self.proteins).isin(group_proteins)
        non_group_proteins = list(pd.Series(self.proteins)[proteins_not_in_group])
        self.proteins = group_proteins + non_group_proteins
        self._reorder_columns()
    
    
    @staticmethod
    def _equation_maker(row: pd.Series) -> str:
        """
        Receives a numerator and denomiator string to produce a single formula used
        to compute biological features.
        :param row: Receives a row of a dataframe containing a numerator with a string 
            which expresses relationship between proteins (eg: A + B or A * B) and 
            possibly a similarly defined denominator. 
        """
        if row["denominator"] is np.nan:
            return row["numerator"]
        else:
            numerator = row["numerator"]
            denominator = row["denominator"]
            if "+" in numerator or "*" in numerator:
                numerator = "(" + numerator + ")"
            if "+" in denominator:
                denominator = "(" + denominator + ")"
            return numerator + " / " + denominator
    
    
    def bio_feature_equations(self, formulas_df: pd.DataFrame):
        """
        It generates a dictionary of biological equations starting from a formulas dataframe.
        :param formulas_df: a df of formulas
        """
        formulas_df.numerator = formulas_df.numerator.str.strip().str.upper()
        formulas_df.denominator = formulas_df.denominator.str.strip().str.upper()
        formulas_df["equation"] = formulas_df.apply(self._equation_maker, 1)
        formulas_df["equation"] = formulas_df.equation.str.replace(" X ", " * ", regex=False)
        formulas_df["equation"] = formulas_df.equation.str.replace("\s+", " ")
        formulas_df = formulas_df[formulas_df.calculate == 1]

        if not self.use_energy_proteins:
            mask = formulas_df.equation.str.contains("|".join(constants.ENERGY_PROTEINS))
            formulas_df = formulas_df[~mask]

        self.formulas_df = formulas_df
        self.equation_mapping = {
            key: val for _, (key, val) in formulas_df[["name", "equation"]].iterrows()
        }
        self.clean_equation_mapping = {
            code: utilities.multiple_replace(formula, constants.EQUATION_SIMPLIFIER)
            for code, formula in self.equation_mapping.items()
        }
        
    
    def bio_features(self, formulas_df: pd.DataFrame):
        self.bio_feature_equations(formulas_df)
        feature_formulas = (self.formulas_df.name + " = " + self.formulas_df.equation).str.cat(sep="\n")
        self.data = self.data.eval(feature_formulas)
    
    
    def feature_names(self):
        return list(self.data.columns[~self.data.columns.isin(self.non_proteins)])

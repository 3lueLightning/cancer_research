import warnings
import pandas as pd

from typing import Optional, List
from typeguard import typechecked
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

#@typechecked
class MinMaxTrimmer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            min_percentile: float=0,
            max_percentile: float=1,
            columns: List[str]=None) -> None:
        assert 0 <= min_percentile <= 1, "min_percentile should be in [0, 1]"
        assert 0 <= max_percentile <= 1, "max_percentile should be in [0, 1]"
        assert min_percentile < max_percentile, "min_percentile should < max_percentile"
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: any=None) -> "MinMaxTrimmer":
        X_select = X[self.columns] if self.columns else X
        self.min_max = X_select.quantile([self.min_percentile, self.max_percentile])
        return self
  
    def transform(self, X: pd.DataFrame, y: any=None) -> pd.DataFrame:
        X_trimmed = X.copy(deep=True)
        cols = self.columns if self.columns else X.columns 
        X_trimmed[cols] = (X_trimmed[cols]
            .clip(self.min_max.iloc[0], self.min_max.iloc[-1], axis=1)
        )
        return X_trimmed

    def fit_tranform(self, X: pd.DataFrame, y: any=None) -> pd.DataFrame:
        return self.fit(X).transform(X)


#@typechecked
class FormulaFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, formula: str):
        self.formula = formula
  
    def fit(self, X: pd.DataFrame, y: any=None) -> 'FormulaFeatureCreator':
        return self
  
    def transform(self, X: pd.DataFrame, y: any=None) -> pd.DataFrame:
        return X.eval(self.formula)

    def fit_transform(self, X: pd.DataFrame, y: any=None) -> pd.DataFrame:
        return self.transform(X)

    def __repr__(self) -> str:
        n_formulas = len(self.formula.split('\n'))
        return f"{self.__class__.__name__}(formulas= {n_formulas} equations)"

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def formula_builder(df: pd.DataFrame) -> str:
        return (df.name + " = " + df.equation).str.cat(sep="\n")


#@typechecked
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: any=None):
        return self
  
    def transform(self, X: pd.DataFrame, y: any=None):
        return X[self.columns]


@typechecked
class MapCategories(BaseEstimator, TransformerMixin):
    """
    Encodes into numbers a column called 'category'.
    Providing a dictionary called mapping allows aggregating multiple categories
    in one.
    Eg. 
        import pandas as pd
        data = pd.Series(["A", "B", "C", "A"])
        map_cat = MapCategories({("A", "C"): "A_C"})
        map_cat.fit_transform(data)
        >> pd.Series([0, 1, 0, 0])
    """
    def __init__(self, mapping:dict):
        self.mapping = mapping
        self.encoder = LabelEncoder() 

    def _map(self, X: pd.Series) -> pd.Series:
        categories = X.copy(deep=True)
        for key, val in self.mapping.items():
            if not isinstance(key, list) and not isinstance(key, tuple):
                key = [key]
            categories[categories.isin(key)] = val
        return categories
  
    def fit(self, X: pd.Series, y: any=None) -> 'MapCategories':
        categories = self._map(X)
        self.encoder.fit(categories)
        self.classes_ = self.encoder.classes_
        return self

    def transform(self, X: pd.Series, y: any=None) -> pd.Series:
        encoded_cats = self.encoder.transform(self._map(X))
        return pd.Series(encoded_cats, name=X.name, index=X.index)

    def fit_transform(self, X: pd.Series, y: any=None) -> pd.Series:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        return pd.Series(self.encoder.inverse_transform(X), name=X.name, index=X.index)


#@typechecked
class PdKBinsDiscretizer(KBinsDiscretizer):
    def __init__(
            self,
            n_bins: int,
            strategy: str="quantile",
            columns: Optional[list]=None) -> None:
        if isinstance(columns, str) and not isinstance(columns, list):
            columns = [columns]
        self.selected_cols = columns
        super().__init__(n_bins=n_bins, encode="ordinal", strategy=strategy)
    
    def fit(self, X: pd.DataFrame, y: any=None) -> 'PdKBinsDiscretizer':
        self.df_columns = X.columns
        if self.selected_cols is None:
            X_selected = X
        else:
            X_selected = X[self.selected_cols]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().fit(X_selected, y)
        return self
    
    def transform(self, X: pd.DataFrame, y: any=None) -> pd.DataFrame:
        if self.selected_cols is None:
            cols = self.df_columns
            X_new = super().transform(X)
            return pd.DataFrame(X_new, columns=cols)
        else:
            X_new = X.copy(deep=True)
            cols = self.selected_cols
            X_new[cols] = super().transform(X[cols])
            return X_new
    
    def fit_transform(self, X: pd.DataFrame, y: any=None) -> pd.DataFrame:
            return self.fit(X, y).transform(X, y)

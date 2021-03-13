import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from typing import List, Type, Optional
from typeguard import typechecked
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (balanced_accuracy_score,
                             accuracy_score,
                             f1_score,
                             confusion_matrix,
                             classification_report)


@typechecked
def plot_confusion(confusion: np.array, category_names: List[str]):
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(confusion, cmap='Blues', annot=True, 
        xticklabels=category_names, yticklabels=category_names)
    ax.set_xlabel('prediction')
    ax.set_ylabel('real value')
    plt.show()


@typechecked
def report_maker(
        estimator: any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        category_names: List[str],
        categories: Optional[List[str]],
        feature_codes: Optional[List[str]],
        equation_mapping: dict,
        weight_array: np.array,
        cv: any = StratifiedKFold(),
        n_features: int = 20,
    ) -> pd.DataFrame:
    """
    Create a report
    """
    print(f'1: {len(estimator["model"].feature_importances_)}')
    estimator = clone(estimator)
    y_train_cv = cross_val_predict(
        estimator,
        X_train,
        y_train,
        cv=cv,
        fit_params={'model__sample_weight': weight_array},
        n_jobs=-1
    )

    estimator.fit(X_train, y_train)
    print(f'2: {len(estimator["model"].feature_importances_)}')
    y_train_pred = estimator.predict(X_train)

    real_v_preds = [
        ["train", y_train, y_train_pred],
        ["cv", y_train, y_train_cv]
    ]

    if X_test is not None:
        y_test_pred = estimator.predict(X_test)
        real_v_preds.append(["test", y_test, y_test_pred])

    print("\n\n")
    for name, real, pred in real_v_preds:
        tree_accuracy = accuracy_score(real, pred)
        tree_balanced_acc = balanced_accuracy_score(
            real,
            pred,
            sample_weight=weight_array
        )

    print(f"{name} f1-macro {f1_score(real, pred, average='macro'):.2%}")
    print(f"{name} f1-micro {f1_score(real, pred, average='micro'):.2%}")
    print(f"{name} accuracy score for best tree: {tree_accuracy:.2%}")
    print(f"{name} balanced accuracy for best tree: {tree_balanced_acc:.2%}")
    print(f"{name} confusion")
    plot_confusion(
      confusion_matrix(real, pred, normalize="pred"),
      category_names
    )

    creport = classification_report(y_train, y_train_cv, 
                                  target_names=category_names)
    print(creport)

    if categories is not None:
        print("recall by category")
        verif = pd.DataFrame({"y": y_train, "pred": y_train_cv,
            "correct": y_train == y_train_cv, "category": categories})
    display(
      pd.DataFrame(
        verif
        .groupby("category")
        .apply(lambda df: df.correct.sum()/ len(df)),
        columns=["recall"]
      ).reset_index()
      .sort_values("recall")
      .style.format({"importance":'{:.2%}'})
    )

    if isinstance(estimator, Pipeline):
        if feature_codes is None:
            feature_codes = estimator["preprocess"]['select_proteins'].columns
        feature_importance = estimator["model"].feature_importances_
    else:
        if feature_codes is None:
            feature_codes = X_train.columns
        feature_importance = estimator.feature_importances_
    feature_names = [equation_mapping.get(code, code) for code in feature_codes]

    feature_importance = pd.DataFrame({
        "code": feature_codes,
        "name": feature_names,
        "importance": feature_importance})
    feature_importance.sort_values("importance", ascending=False, inplace=True)

    print("\n\ndecision tree feature importance")
    with pd.option_context('display.max_colwidth', 400):
        display(
            feature_importance
            .loc[feature_importance.importance > 0, ["name","importance"]]
            .reset_index(drop=True)
            .head(n_features)
            .style.format({"importance":'{:.2%}'})
        )
  
    # plot feature importance
    print("importance of each individual feature (ordered)")
    ax = (feature_importance.reset_index().importance * 100).plot.line()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()
    print("cumulative importance of each feature (ordered)")
    ax = (feature_importance.reset_index().importance.cumsum() * 100).plot.line()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()

    return feature_importance


@typechecked
def plot_protein_heatmap(
        data: pd.DataFrame,
        proteins: pd.Series,
        category_mapping: Optional[dict]=None,
        feature_mapping: Optional[dict]=None,
        group_proteins: bool=True
    ) -> None:
    """
    Create a boxplot and a heatmap of protein expression levels accross categories
    """
    
    if group_proteins:
        cols = protein_group.Protein[ protein_group.Protein.isin(proteins)]
    else:
        cols = proteins
        
    ss_data = StandardScaler().fit_transform(data[cols])
    ss_data = pd.DataFrame(ss_data, columns=cols)
    
    ss_data["category"] = data["category"].astype("category")
    if category_mapping:
        sorter = list(category_mapping.values())
        category_encoder = MapCategories( category_mapping)
        encoded_categories = category_encoder.fit_transform( data.category)
        ss_data["category"] = category_encoder.inverse_transform( encoded_categories)
        ss_data["category"].cat.set_categories(sorter, inplace=True)
    ss_data.sort_values("category", inplace=True)
    
    total_deviation = (ss_data[cols] - ss_data[cols].min()).sum(axis="columns").rename("dev")
    ss_data = pd.concat([ss_data, total_deviation], 1).sort_values(["category", "dev"])[ss_data.columns]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    dev_by_category = pd.concat([total_deviation, ss_data.category], 1)
    sns.boxplot(data=dev_by_category, x="dev", y="category")
    ax.axvline(dev_by_category[dev_by_category.category == "CTL"].dev.median())
    plt.xlabel("sum deviations from the minimum")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(30,20))
    ss_data = ss_data[cols]
    if feature_mapping:
        ss_data.rename(columns=feature_mapping, inplace=True)
    thresh = np.quantile(ss_data.values.reshape(-1), q=.98)
    hm = sns.heatmap( ss_data[cols], vmax=thresh, cmap="YlGnBu", yticklabels=[])
    for _, (category, val) in ss_data.category.value_counts(sort=False).cumsum().reset_index().iterrows():
        if val < len(ss_data):
            ax.axhline(val, ls='-', color="black", linewidth=3)
        ax.text(48, val - 5, category, fontsize=15)

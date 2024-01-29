from copy import deepcopy
from random import Random

import imblearn
import pandas as pd
from sklearn.base import ClassifierMixin
<<<<<<< HEAD
import mlflow

from .metric_calculation import get_metrics

mlflow.sklearn.autolog()

=======

from .metric_calculation import get_metrics

>>>>>>> main

def create_folds(values: list, n_folds: int, seed: int) -> dict:
    """
    This function creates n_folds amount of folds from values, each fold is a dictionary with a test and a train key,
    where 'test' and 'train' never overlap and the test sets are different for every fold.

    @param values: list of values
    @param n_folds: number of folds to make
    @param seed: random seed for shuffling data

    @return dictionary with integer keys, the value of each key is a dictionary with keys 'train' and 'test', each value of those is a subset of values.
    """
    values_shuffled = values.copy()
    Random(seed).shuffle(values_shuffled)
    hold_out_per_fold = len(values) // n_folds
    folds = {}
    for n in range(0, n_folds):
        hold_out_values = values_shuffled[
            hold_out_per_fold * n : hold_out_per_fold * (n + 1)
        ]
        folds[n] = {
            "train": [value for value in values if value not in hold_out_values],
            "test": hold_out_values,
        }
    return folds


def train_imbalanced_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: ClassifierMixin,
    random_state: int,
    sampling_type_boundary: int,
):
    """
    Rebalances X_train and y_train (oversampling or undersampling depending on the size of the smallest class) and then fits model. Does not return
    anything as model is fit in place.

    @param X_train: dataframe of features to train model on
    @param y_train: series of label to train model on
    @param model: initialised classifier model
    @param random_state: seed for rebalancing
    @sampling_type_boundary: if size of smallest class < sampling_type_boundary we oversample, otherwise we undersample
    """

    size_smallest_class = y_train.value_counts().min()
    if size_smallest_class < sampling_type_boundary:
        print("Oversampling to rebalancing dataset")
        sampler = imblearn.over_sampling.SMOTE(random_state=random_state)
    else:
        print("Undersampling to rebalancing dataset")
        sampler = imblearn.under_sampling.RandomUnderSampler(random_state=random_state)
    X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)

    print("Fitting model")
<<<<<<< HEAD

    with mlflow.start_run() as run:
        model.fit(X_balanced, y_balanced)
=======
    model.fit(X_balanced, y_balanced)
>>>>>>> main


def cross_validation_balance_on_date(
    data: pd.DataFrame,
    model: ClassifierMixin,
    cv: int,
    features: list,
    random_state: int,
    sampling_type_boundary: int = 100000,
) -> list:
    """
    This method does cross validation based on dates instead of sampling.

    @param data: pandas DataFrame with a date column.
    @param model: a model with making predictions.
    @param cv: the number of folds.
    @param features: list of columns in data to use
    @param random_state: seed for cross_validation split and balancing imbalanced datasets
    @param sampling_type_boundary: if size of smallest class < sampling_type_boundary we oversample, otherwise we undersample

    @return list of dictionaries. Each dictionary has a key 'fold': integer corresponding to fold, 'train': train metrics, 'test': test metrics,
            'model': trained model and 'hold_out_dates': dates that were in test for this fold
    """

    results = []

    folds = create_folds(list(data["date"].unique()), cv, random_state)

    for fold in folds.keys():
        print("---------fold: " + str(fold + 1))

        print("Picked hold out dates: ")
        print(folds[fold]["test"])

        df_train = data[data["date"].isin(folds[fold]["train"])]
        df_test = data[data["date"].isin(folds[fold]["test"])]

        train_imbalanced_model(
            X_train=df_train[features],
            y_train=df_train["label"],
            model=model,
            random_state=random_state,
            sampling_type_boundary=sampling_type_boundary,
        )

        print("Calculating train metrics")
        train = get_metrics(y=df_train["label"], X=df_train[features], model=model)
        print("Calculating test metrics")
        test = get_metrics(y=df_test["label"], X=df_test[features], model=model)
        results.append(
            {
                "fold": fold + 1,
                "train": train,
                "test": test,
                "model": deepcopy(model),
                "hold_out_dates": folds[fold]["test"],
            }
        )

    return results

import pandas as pd
from sklearn import metrics
from sklearn.base import ClassifierMixin, TransformerMixin


def get_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    model: ClassifierMixin,
    scaler: TransformerMixin = None,
) -> dict:
    """
    Get precision, recall, f1-score and support per class for X, y and model.
    (not using sklearn.metrics.classification_report, because it is considerably more computationally intensive)

    @param X: dataframe of features as necessary for model
    @param y: true values for the labels of X dataframe
    @param model: Classifier that predicts y based on X
    @param scaler: sklearn scaler that will be used if passed as argument

    @return dictionary with keys of class names, each value is a dictionary of 'precision', 'recall', 'f1-score' and 'support'
    """
    if scaler is not None:
        X = scaler.transform(X)
    confusion_matrix = metrics.confusion_matrix(y, model.predict(X))
    metrics_dict = {}
    for i in range(0, len(confusion_matrix)):
        precision = confusion_matrix[i, i] / confusion_matrix[:, i].sum()
        recall = confusion_matrix[i, i] / confusion_matrix[i, :].sum()
        f1_score = 2 * precision * recall / (precision + recall)
        support = confusion_matrix[i, :].sum()
        metrics_dict[model.classes_[i]] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": support,
        }
    return metrics_dict


def calculate_average_metrics(results: list) -> pd.DataFrame:
    """
    Transforms results dictionary (as output from cross_validation_balance_on_date) and transforms it into precision, recall, f1-score per label averaged over all folds.

    @param results: list of fold dictionaries, as per output of cross_validation_balance_on_date

    @return Dataframe with labels as index and averaged precision, recall and f1-score as columns.
    """
    dfs = []
    for i in range(0, len(results)):
        df = pd.DataFrame.from_dict(results[i]["test"]).transpose()
        df["fold"] = results[i]["fold"]
        df[["precision", "recall", "f1-score"]] = df[
            ["precision", "recall", "f1-score"]
        ].multiply(df["support"], axis="index")
        dfs += [df]

    df_folds = pd.concat(dfs)
    df_grouped = df_folds.groupby(df_folds.index)
    df_avg_folds = df_grouped.sum()
    df_avg_folds = df_avg_folds.div(df_grouped["support"].sum(), axis="index")

    return df_avg_folds[["precision", "recall", "f1-score"]]

"""
Common evaluation metrics for both forecasting and NLP experiments.
All functions accept numpy arrays and return float values.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error. Skips zeros in y_true to avoid division by zero."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def all_forecasting_metrics(y_true, y_pred):
    """Return dict with MAPE, RMSE, MAE for a forecasting experiment."""
    return {
        "MAPE (%)": round(mape(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 4)
    }


def all_classification_metrics(y_true, y_pred, labels=None):
    """Return dict with Accuracy, macro F1, Precision, Recall for classification."""
    return {
        "Accuracy (%)": round(accuracy_score(y_true, y_pred) * 100, 2),
        "F1 (macro)": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Precision (macro)": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Recall (macro)": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4)
    }


def get_classification_report(y_true, y_pred, labels=None):
    """Return sklearn classification report as a formatted string."""
    return classification_report(y_true, y_pred, target_names=labels, zero_division=0)


def get_confusion_matrix(y_true, y_pred):
    """Return confusion matrix as a 2D numpy array."""
    return confusion_matrix(y_true, y_pred)

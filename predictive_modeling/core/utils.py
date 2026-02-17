"""
Utilities for tuning probability decision thresholds
based on the F1-score of validation data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Literal

from sklearn.metrics import f1_score


def find_best_threshold_for_death(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    death_label: int = 0,
    grid: int = 101,
) -> Tuple[float, float]:
    """
    Find the probability threshold that maximizes F1 for the 'Died' class based on validation data.

    Args:
        model: trained classifier with predict_proba
        X_val: validation features
        y_val: true labels (0/1)
        death_label: which label represents "Died" (default=0)
        grid: number of threshold points between 0 and 1

    Returns:
        best_thresh: float
        best_f1: float
    """

    # Probability of death (class 0 by default)
    y_proba = model.predict_proba(X_val)[:, death_label]

    thresholds = np.linspace(0, 1, grid)
    f1_scores = []

    for t in thresholds:
        # Apply threshold
        y_tmp = (
            (y_proba >= t).astype(int) * death_label
            + (y_proba < t).astype(int) * (1 - death_label)
        )

        f1_scores.append(
            f1_score(y_val, y_tmp, pos_label=death_label)
        )

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def find_best_threshold_overall(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    average: Literal["weighted", "macro", "micro"] = "weighted",
    grid: int = 101,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes the overall F1-score on validation data.

    Args:
        model: trained classifier with predict_proba
        X_val: validation features
        y_val: true labels (0/1)
        average: "weighted", "macro", or "micro"
        grid: number of threshold points between 0 and 1

    Returns:
        best_thresh: float
        best_f1: float
    """

    # Probability of class=1 (Recovered/Normal)
    y_proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0, 1, grid)
    f1_scores = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1_scores.append(
            f1_score(y_val, y_pred, average=average)
        )

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

"""
All class-imbalance handling lives here:
- undersampling
- oversampling
- SMOTEENN
- ADASYN + Tomek
- no sampling
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional, Tuple

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbalancePipeline


def resample_dataset(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    weights_train: Optional[pd.Series] = None,
    method: str = "undersample",
    use_weights: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Resample training dataset with undersampling, oversampling, both, or none.

    Parameters
    ----------
    x_train, y_train : pd.DataFrame, pd.Series
        Training features and labels.
    x_test, y_test : pd.DataFrame, pd.Series
        Test features and labels.
    weights_train, weights_test : array-like or None
        Sample weights for training and test sets (optional).
    method : str
        One of:
        - "undersample"
        - "oversample"
        - "smoteenn"
        - "adasyn_tomek"
        - "nosample"
    use_weights : bool
        Whether to keep original sample weights.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    x_train_final, y_train_final, weights_train_final
    """

    # Combine X and y for easier processing
    train_df = x_train.copy()
    train_df["y"] = y_train

    if method in ["undersample", "oversample", "smoteenn", "adasyn_tomek"]:

        # Choose sampler
        if method == "undersample":
            sampler = RandomUnderSampler(
                sampling_strategy="majority",
                random_state=random_state
            )

        elif method == "oversample":
            sampler = RandomOverSampler(
                sampling_strategy="minority",
                random_state=random_state
            )

        elif method == "smoteenn":
            sampler = SMOTEENN(random_state=random_state)

        elif method == "adasyn_tomek":
            sampler = ImbalancePipeline([
                ("adasyn", ADASYN(random_state=random_state)),
                ("tomek", TomekLinks())
            ])

        # Apply resampling
        X_resampled, y_resampled = sampler.fit_resample(
            train_df.drop(columns=["y"]),
            train_df["y"]
        )

        # Rebuild DataFrame
        resampled_df = pd.DataFrame(
            X_resampled,
            columns=x_train.columns
        )
        resampled_df["y"] = y_resampled

        if use_weights and weights_train is not None:
            # Merge original weights back
            resampled_with_weights = resampled_df.merge(
                train_df.assign(weight=weights_train),
                how="left",
                on=list(x_train.columns) + ["y"]
            )

            weights_train_final = (
                resampled_with_weights["weight"]
                .fillna(1.0)
            )

            x_train_final = (
                resampled_with_weights
                .drop(columns=["y", "weight"])
            )

            y_train_final = (
                resampled_with_weights["y"]
            )

        else:
            x_train_final = resampled_df.drop(columns=["y"])
            y_train_final = resampled_df["y"]
            weights_train_final = None

    elif method == "nosample":
        return x_train, y_train, weights_train

    else:
        raise ValueError("method must be one of: 'undersample', 'oversample', 'smoteenn', 'adasyn_tomek', or 'nosample'"
        ) 

    # print(f"\nMethod sampling: {method} | use_weights={use_weights}")
    # print("✅ Resampled class distribution:", Counter(y_train_final))
    # print(
    #     "Training set shape:",
    #     x_train_final.shape,
    #     y_train_final.shape,
    #     weights_train_final.shape
    #     if weights_train_final is not None else None,
    # )

    return x_train_final, y_train_final, weights_train_final


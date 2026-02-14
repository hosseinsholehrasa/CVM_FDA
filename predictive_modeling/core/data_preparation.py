"""
All data loading, cleaning, aggregation, encoding, and
train/val/test splitting lives here.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(cleaned_path: str, pubchem_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw data + PubChem features and apply basic cleaning.

    Returns:
        merged dataframe ready for ML processing
    """

    fda_df = pd.read_csv(cleaned_path, low_memory=False)
    pubchem_df = pd.read_csv(pubchem_path, low_memory=False)

    # Drop cases with missing CID
    bad_ids = fda_df.loc[fda_df['CID'].isna(), 'unique_aer_id_number'].unique()
    fda_df = fda_df[~fda_df['unique_aer_id_number'].isin(bad_ids)]

    # Drop unused columns
    fda_df = fda_df.drop(columns=['year', 'dose'], errors='ignore')

    # Harmonize label
    fda_df['medical_status'] = fda_df['medical_status'].replace(
        'Recovered with Sequela', 'Recovered/Normal'
    )

    # Keep only selected PubChem columns
    pubchem_df = pubchem_df[[
        'CID', 'Molecular Weight', 'Hydrogen Bond Donor Count', 'Hydrogen Bond Acceptor Count', 'Rotatable Bond Count', 
        'Exact Mass', 'Monoisotopic Mass', 'Topological Polar Surface Area', 'Heavy Atom Count', 
        'Formal Charge', 'Complexity', 'Isotope Atom Count', 'Defined Atom Stereocenter Count', 
        'Undefined Atom Stereocenter Count', 'Defined Bond Stereocenter Count', 'Undefined Bond Stereocenter Count',
        'Covalently-Bonded Unit Count','Compound Is Canonicalized'
    ]]

    # Merge PubChem features into main FDA dataframe
    model_df = fda_df.merge(pubchem_df, on='CID', how='left')

    # Convert TPSA to numeric
    if 'Topological Polar Surface Area' in model_df.columns:
        model_df['Topological Polar Surface Area'] = (
            model_df['Topological Polar Surface Area']
            .str.replace(" Å²", "", regex=False)
        )
        model_df['Topological Polar Surface Area'] = pd.to_numeric(
            model_df['Topological Polar Surface Area'], errors='coerce'
        )

    # Drop IDs not used for modeling
    model_df = model_df.drop(columns=['CID', 'drug_id', 'atc_vet_code'], errors='ignore')

    return model_df, fda_df, pubchem_df


def aggregate_by_id(
    model_df: pd.DataFrame,
    numeric_columns: List[str] = [],
    sum_columns: List[str] = [],
    categorical_columns: List[str] = [],
    drop_columns: List[str] = []
) -> pd.DataFrame:
    """
    Aggregate data at the unique_aer_id_number level.
    """
    
    def join_unique(x):
        return ' / '.join(sorted(set(str(v) for v in x if pd.notna(v))))

    agg_dict = {
        **{col: 'mean' for col in numeric_columns},
        **{col: 'sum' for col in sum_columns},
        **{col: join_unique for col in categorical_columns}
    }

    aggregated = (
        model_df
        .groupby('unique_aer_id_number')
        .agg(agg_dict)
        .reset_index()
    )

    aggregated = aggregated.drop(columns=[c for c in drop_columns if c in aggregated.columns])

    return aggregated


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Fit LabelEncoders on categorical columns.
    Returns transformed dataframe + encoders.
    """

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


def build_labeled_unlabeled_sets(
    X: pd.DataFrame,
    Y: pd.DataFrame
):
    """
    Separate labeled (Death and Recovered/Normal) vs unlabeled data (Ongoing and Outcome Unknown).
    """
    label_map = {0: "Died", 1: "Recovered/Normal"}
    reversed_label_map = {v: k for k, v in label_map.items()}

    # Filter only 'Died' and 'Recovered/Normal' Labeled cases (exclude Ongoing + Outcome Unknown)
    X_labeled = X[Y['Ongoing'] == 0]
    Y_labeled = Y[Y['Ongoing'] == 0]

    X_labeled = X_labeled[Y_labeled['Outcome Unknown'] == 0]
    Y_labeled = Y_labeled[Y_labeled['Outcome Unknown'] == 0]

    # Extract number of animals as weights of each report
    weights_labeled = X_labeled['number_of_animals'].astype(float)
    X_labeled = X_labeled.drop(columns=['number_of_animals'])
    Y_labeled = Y_labeled.drop(columns=['Outcome Unknown', 'Ongoing'])

    # Convert Y to 1D label (0 or 1)
    Y_labeled = Y_labeled.idxmax(axis=1)    # "Died" or "Recovered/Normal"
    Y_labeled = Y_labeled.map(reversed_label_map)

    # Unlabeled pool (Ongoing and Outcome Unknown cases)
    X_unlabeled = pd.concat([
        X[Y['Ongoing'] == 1],
        X[Y['Outcome Unknown'] == 1]
    ])
    Y_unlabeled = pd.concat([Y[Y['Ongoing'] == 1], Y[Y['Outcome Unknown'] == 1]])

    weights_unlabeled = X_unlabeled['number_of_animals'].astype(float)
    X_unlabeled = X_unlabeled.drop(columns=['number_of_animals'])
    Y_unlabeled = Y_unlabeled.drop(columns=['Outcome Unknown', 'Ongoing'])

    print("Labeled data shape:", X_labeled.shape, Y_labeled.shape)
    print("Unlabeled data shape:", X_unlabeled.shape, Y_unlabeled.shape)

    return (
        X_labeled, Y_labeled, weights_labeled,
        X_unlabeled, Y_unlabeled, weights_unlabeled
    )


def stratified_split(
    X_labeled: pd.DataFrame,
    Y_labeled: pd.Series,
    weights_labeled: pd.Series,
    test_size: float = 0.1,
    random_state: int = 42
):
    """
    Perform 80/10/10 stratified split.
    """

    x_train_idx, x_temp_idx = train_test_split(
        X_labeled.index,
        stratify=Y_labeled,
        test_size=test_size * 2,  # because we will split temp into val and test
        random_state=random_state
    )

    x_val_idx, x_test_idx = train_test_split(
        x_temp_idx,
        stratify=Y_labeled.loc[x_temp_idx],
        test_size=0.5,
        random_state=random_state
    )

    # Assign splits
    x_train = X_labeled.loc[x_train_idx]
    y_train = Y_labeled.loc[x_train_idx]
    w_train = weights_labeled.loc[x_train_idx]

    x_val = X_labeled.loc[x_val_idx]
    y_val = Y_labeled.loc[x_val_idx]
    w_val = weights_labeled.loc[x_val_idx]

    x_test = X_labeled.loc[x_test_idx]
    y_test = Y_labeled.loc[x_test_idx]
    w_test = weights_labeled.loc[x_test_idx]

    print("Training set:", x_train.shape, y_train.shape, w_train.shape)
    print("Validation set:", x_val.shape, y_val.shape, w_val.shape)
    print("Test set:", x_test.shape, y_test.shape, w_test.shape)

    return x_train, y_train, w_train, x_val, y_val, w_val, x_test, y_test, w_test


def finalize_dataset(
    x: pd.DataFrame,
    y: pd.Series,
    weights: Optional[pd.Series],
    ):
    """
    Convert the dataset ready for model training in numpy format.
    """

    x_final = x.to_numpy()
    y_final = y.to_numpy()
    weights_final = (
        weights.to_numpy()
        if weights is not None else None
    )

    return x_final, y_final, weights_final

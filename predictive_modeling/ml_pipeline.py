from typing import Counter, Optional
import pandas as pd
import numpy as np
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from predictive_modeling.core.data_preparation import (
    load_and_clean_data,
    aggregate_by_id,
    encode_categoricals,
    build_labeled_unlabeled_sets,
    stratified_split,
    finalize_dataset,
)
from predictive_modeling.core.sampling import resample_dataset
from predictive_modeling.ml.supervised import train_and_evaluate
from predictive_modeling.ml.semi_supervised import semi_supervised_training
from predictive_modeling.ml.explainability import (
    fda_shap_summary_plot,
    fda_plot_top_bottom_shap,
)


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.core")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


RANDOM_STATE = 42


def load_models_config():
        
    models_config = {
        "logistic": LogisticRegression(max_iter=1000, n_jobs=-1),
        "decision_tree": DecisionTreeClassifier(),
        "knn": KNeighborsClassifier(n_jobs=-1),
        "random_forest": RandomForestClassifier(verbose=0, n_jobs=-1),
        "adaboost": AdaBoostClassifier(),
        "catboost": cb.CatBoostClassifier(verbose=0),
        "xgboost": xgb.XGBClassifier(tree_method="hist", device="cuda", random_state=RANDOM_STATE),  # nthread=-1
        "lda": LinearDiscriminantAnalysis(),
        "mlp": MLPClassifier(),   # takes more time
        "gradient_boosting": GradientBoostingClassifier(),  # takes more time
        "voting": VotingClassifier(
            estimators=[
                ("lr", RandomForestClassifier(verbose=0, n_jobs=-1)),
                ("rf", cb.CatBoostClassifier(verbose=0)),
                ("gb", xgb.XGBClassifier(tree_method="hist", device="cuda", random_state=RANDOM_STATE))
            ],
            voting="soft"
        ),
        "stacking": StackingClassifier(
            estimators=[
                ("lr", RandomForestClassifier(verbose=0, n_jobs=-1)),
                ("rf", cb.CatBoostClassifier(verbose=0)),
                ("gb", xgb.XGBClassifier(tree_method="hist", device="cuda", random_state=RANDOM_STATE))
            ],
            final_estimator=LogisticRegression()
        )
    }

    return models_config


def ml_pipeline(config: Optional[dict] = None):

    print("🚀 Starting ML Pipeline...")

    ############### data preparation ######################
    print("📂 Loading and cleaning data...")
    model_df, fda_df, pubchem_df = load_and_clean_data("data/expanded_cleaned_data.csv",
                                                 "data/pubchem_compounds.csv")

    # Define columns for aggregation and encoding
    numeric_columns = ['animal_age', 'animal_weight', 'number_of_animals']
    sum_columns = pubchem_df.columns.difference(['CID', 'Compound Is Canonicalized']).tolist()
    categorical_label_features = [
        'animal_species',
        'animal.gender',
        'animal.breed.breed_component',
        'Compound Is Canonicalized',
        'name',
        'dosage_form',
        'route',
        'veddra_term_name',
        'medical_status'
    ]
    # Drop redundant / collinear chemistry features
    drop_columns = [
        'Complexity',
        'Defined Atom Stereocenter Count',
        'Exact Mass',
        'Heavy Atom Count',
        'Hydrogen Bond Acceptor Count',
        'Monoisotopic Mass',
        'Rotatable Bond Count',
        'Topological Polar Surface Area',
        'Hydrogen Bond Donor Count',
    ]

    print("🔄 Aggregating and encoding data...")
    aggregated_model_df = aggregate_by_id(
        model_df,
        numeric_columns=numeric_columns,
        sum_columns=sum_columns,
        categorical_columns=categorical_label_features,
        drop_columns=drop_columns
    )

    categorical_features_encoder = [c for c in categorical_label_features if c != "medical_status"]
    dataset, label_encoders = encode_categoricals(aggregated_model_df, categorical_features_encoder)

    print()
    print("🔄 Building labeled and unlabeled sets...")
    X = dataset.drop(columns=['unique_aer_id_number','medical_status'])
    Y = pd.get_dummies(dataset['medical_status'])

    X_labeled, Y_labeled, w_labeled, X_unlabeled, Y_unlabeled, w_unlabeled = build_labeled_unlabeled_sets(X, Y)

    print()
    print("🔄 Stratified splitting into train/val/test...")
    x_train, y_train, w_train, x_val, y_val, w_val, x_test, y_test, w_test = \
        stratified_split(X_labeled, Y_labeled, w_labeled, random_state=RANDOM_STATE)



    ############### Training configs, Supervised Training and Evaluation ######################
    print()
    print("🚀 Starting training and evaluation...")
    print()
    models_config = load_models_config()
    
    # "nosample", "undersample", "oversample", "smoteenn", "adasyn_tomek"
    training_configs = [
        {"name": "nosample + no weights", "method": "nosample", "use_weights": False},
        {"name": "undersample + no weights", "method": "undersample", "use_weights": False},
        {"name": "oversample + no weights", "method": "oversample", "use_weights": False},
        {"name": "SMOTEENN + no weights", "method": "smoteenn", "use_weights": False},
        {"name": "adasyn tomek + no weights", "method": "adasyn_tomek", "use_weights": False},
    ]

    ml_results = []
    for cfg in training_configs:
        models_list = [
            "logistic", "decision_tree", "knn", "random_forest", "adaboost", "catboost", 
            "xgboost", "voting", "stacking", "mlp"
        ]

        for model_name in models_list:
            print("="*80)
            print(f"Supervised training: {cfg['name']}, model: {model_name}")

            # Call train_and_evaluate
            result = train_and_evaluate(
                x_train, y_train, x_test, y_test, x_val, y_val,
                w_train if cfg["use_weights"] else None,
                w_test if cfg["use_weights"] else None,
                w_val if cfg["use_weights"] else None,
                method=cfg["method"], use_weights=cfg["use_weights"],
                random_state=RANDOM_STATE,
                model_config=models_config[model_name]
            )

            # Get F1 score from returned metrics
            f1 = result["metrics"]["f1"]
            print(f"F1 score: {f1:.3f}")

            ml_results.append({"Setting": cfg["name"], "F1_score": f1, "classification_report": result["metrics"]["classification_report"], "details": result, "model_name": model_name})

            # Convert results to DataFrame for ranking
            df_ml_results = pd.DataFrame(ml_results).sort_values(by="F1_score", ascending=False)

    print()
    print("*"*80)
    # Convert results to DataFrame for ranking
    df_ml_results = pd.DataFrame(ml_results).sort_values(by="F1_score", ascending=False)

    print("📊 All Results:")
    print(df_ml_results[["Setting", 'model_name', "F1_score"]])

    best_ml_row = df_ml_results.iloc[0]
    print(f"\n✅ Best setting: {best_ml_row['Setting']} with F1 = {best_ml_row['F1_score']:.3f}")
    print("*"*80)
    print("\n\n")

    ############### Semi-supervised configs, Semi-supervised training and Evaluation ######################
    
    # "aum", "prob", "entropy"
    # "nosample", "undersample", "oversample", "smoteenn", "adasyn_tomek"
    semi_configs = [
        {"name": "nosample + no weights (top_k=0.3, aum)", "resample_method": "nosample", "top_k": 0.3, "use_weights": False, "confidence_method": "aum"},
        {"name": "undersample + no weights (top_k=0.3, aum)", "resample_method": "undersample", "top_k": 0.3, "use_weights": False,  "confidence_method": "prob"},
        {"name": "oversample + no weights (top_k=0.3, aum)", "resample_method": "oversample", "top_k": 0.3, "use_weights": False, "confidence_method": "aum"},
        {"name": "SMOTEENN + no weights (top_k=0.3, aum)",       "resample_method": "smoteenn",      "top_k": 0.3, "use_weights": False,  "confidence_method": "aum"},
        {"name": "adasyn tomek + no weights (top_k=0.3, aum)",       "resample_method": "adasyn_tomek",      "top_k": 0.3, "use_weights": False,  "confidence_method": "aum"},
    ]

    semi_results = []

    for cfg in semi_configs:
        print()
        print("+"*80)
        print(f"Semi-supervised training: {cfg['name']}")
        # Find all supervised results that match the resample/weight setting
        supervised_matches = df_ml_results[
            (df_ml_results["Setting"].str.contains(cfg["resample_method"].replace("_", " "), case=False)) &
            (df_ml_results["Setting"].str.contains("no weights" if not cfg["use_weights"] else "weights", case=False))
        ]

        # print(supervised_matches.iloc[0]['Setting'])
        for _, sup_row in supervised_matches.iterrows():
            base_model = sup_row['details']["model"]
            model_name = sup_row["model_name"]
            # print(f"   Base model: {model_name}")
            print("-"*80)
            print(f"Model: {model_name}")

            semi_result = semi_supervised_training(
                base_model,
                X, Y,
                x_train, y_train, w_train if cfg["use_weights"] else None,
                x_test, y_test, w_test if cfg["use_weights"] else None,
                x_val, y_val, w_val if cfg["use_weights"] else None,
                confidence_method=cfg["confidence_method"],
                top_k_percent=cfg["top_k"],
                resample_method=cfg["resample_method"],
                use_weights=cfg["use_weights"],
                num_passes=5
            )

            f1_semi = semi_result["metrics"]["f1"]
            print(f"F1 score: {f1_semi:.3f}")

            semi_results.append({
                "Setting": cfg["name"],
                "Supervised_Setting": sup_row["Setting"],
                "Supervised_Model": model_name,
                "F1_score": f1_semi,
                "classification_report": semi_result['metrics']['classification_report'],
                "details": semi_result
            })
            df_semi_results = pd.DataFrame(semi_results).sort_values(by="F1_score", ascending=False)

    df_semi_results = pd.DataFrame(semi_results).sort_values(by="F1_score", ascending=False)
    
    print()
    print("*"*80)
    print("📊 All Results:")
    print(df_semi_results[["Setting", 'Supervised_Model', "F1_score"]])
    print(f"\n✅ Best setting: {df_semi_results.iloc[0]['Setting']} with F1 = {df_semi_results.iloc[0]['F1_score']:.3f}, Model = {df_semi_results.iloc[0]['Supervised_Model']}")
    print("*"*80)
    semi_model = df_semi_results.iloc[0]['details']['model']
    print("\n\n")

    ############### SHAP ######################
    shap_input = x_test.copy()
    shap_cleaned_feature_names = (
        X_labeled.columns.to_series()
        .str.replace('animal.breed.breed_component', "Breed", regex=True)
        .str.replace("animal.", "", regex=True)
        .str.replace("veddra_term_name", "AEs", regex=False)
        .str.replace('route', 'Administration Route', regex=False)
        .str.replace("_", " ", regex=False)
        .str.replace("^name$", "Active Ingredient Name", regex=True)
        .tolist()
    )

    species_le = label_encoders['animal_species']
    species_groups = {
        "companion": ["Cat", "Dog", "Rabbit"],
        "poultry": ["Chicken", "Turkey"],
        "livestock": ["Cattle", "Sheep", "Goat", "Pig", "Horse"]
    }

    print("\n🚀 Generating SHAP explainability plots for top semi-supervised model in different animal groups...")
    for animal_group_name, species_names in species_groups.items():

        labels = species_le.transform(species_names)
        shap_subset = shap_input[
            shap_input['animal_species'].isin(labels)
        ].copy()

        # plot and save the shap summary, active ingredients, and adverse events
        shap_values = fda_shap_summary_plot(
            model=semi_model,
            shap_input=shap_subset,
            cleaned_feature_names=shap_cleaned_feature_names,
            save_path=f"results/shap_summary_{animal_group_name}.png",
            max_display=15)

        fda_plot_top_bottom_shap(
            shap_values=shap_values,
            x_test=shap_subset,
            field_name="veddra_term_name",
            label_encoders=label_encoders,
            min_samples=5,
            top_n=5,
            y_axis_label="Adverse Events",
            save_path=f"results/AEs_{animal_group_name}.png"
        )
        
        fda_plot_top_bottom_shap(
            shap_values=shap_values,
            x_test=shap_subset,
            field_name="name",
            label_encoders=label_encoders,
            min_samples=5,
            top_n=5,
            y_axis_label="Active Ingredient Names",
            save_path=f"results/ingredients_{animal_group_name}.png"
        )


if __name__ == "__main__":
    ml_pipeline()

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_fscore_support,
    balanced_accuracy_score
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

from predictive_modeling.core.data_preparation import finalize_dataset
from predictive_modeling.core.sampling import resample_dataset
from predictive_modeling.core.utils import find_best_threshold_for_death


def train_and_evaluate(
    x_train, y_train, 
    x_test, y_test, 
    x_val, y_val, 
    w_test=None, w_train=None, w_val=None,
    method="undersample", use_weights=False, random_state=42,
    model_config=xgb.XGBClassifier(tree_method="hist", device="cuda", random_state=42)
):
    """
    Train ML models with optional sampling and weights, evaluate with metrics.

    Returns
    -------
    results: dict
        {
            "model": trained model,
            "y_pred": predictions,
            "x_train": final training features,
            "y_train": final training labels,
            "x_test": final test features,
            "y_test": final test labels,
            "w_train": final training weights,
            "w_test": final test weights,
            "x_val": final validation features,
            "y_val": final validation labels,
            "w_val": final validation weights,
            "metrics": {accuracy, balanced_accuracy, f1, confusion_matrix, classification_report}
        }
    """
    label_map = {0: "Died", 1: "Recovered/Normal"}

    # Resample dataset
    x_train_sampled, y_train_sampled, w_train_sampled = resample_dataset(
        x_train,
        y_train,
        w_train,
        method=method,
        use_weights=use_weights,
        random_state=random_state
    )

    # finalized training ready for modeling
    x_train_final, y_train_final, w_train_final = finalize_dataset(
        x_train_sampled,
        y_train_sampled,
        w_train_sampled,
    )
    x_test_final, y_test_final, w_test_final = finalize_dataset(
        x_test,
        y_test,
        w_test,
    )
    x_val_final, y_val_final, w_val_final = finalize_dataset(
        x_val,
        y_val,
        w_val,
    )

    # print("\n✅ Final training class distribution:", Counter(y_train_final))

    # Define Model Pipeline
    best_model_pipeline = Pipeline(steps=[
        # ("pca", PCA(n_components=20, random_state=random_state)),
        ('classifier', model_config),
    ], verbose=False)

    # Train
    if use_weights and w_train_final is not None:
        best_model_pipeline.fit(
            x_train_final, y_train_final,
            classifier__sample_weight=w_train_final
        )
    else:
        best_model_pipeline.fit(x_train_final, y_train_final)


    # Predict
    # y_pred_final = best_model_pipeline.predict(x_test_final)
    # thresholds = np.linspace(0, 1, 101)
    # y_proba_final = best_model_pipeline.predict_proba(x_test_final)[:, 0]

    # f1_scores = []
    # for t in thresholds:
    #     y_tmp = (y_proba_final >= t).astype(int)
    #     f1_scores.append(f1_score(y_test_final, y_tmp, pos_label=0))
    # best_thresh = thresholds[int(np.argmax(f1_scores))]
    # y_pred_final = (y_proba_final >= best_thresh).astype(int)

    # Tune threshold on validation set
    death_label = 0
    best_thresh, best_f1 = find_best_threshold_for_death(best_model_pipeline, x_val, y_val, death_label=death_label)
    # print(f"\n🔎 Best threshold from validation = {best_thresh:.2f}, F1 (Died={death_label}) = {best_f1:.3f}")

    # Apply threshold on test set
    y_proba_final = best_model_pipeline.predict_proba(x_test_final)[:, death_label]
    y_pred_final = (y_proba_final >= best_thresh).astype(int) * death_label + (y_proba_final < best_thresh).astype(int) * (1 - death_label)


    # Evaluation
    if use_weights and w_test_final is not None:
        acc = accuracy_score(y_test_final, y_pred_final, sample_weight=w_test_final)
        f1 = f1_score(y_test_final, y_pred_final, average='weighted', sample_weight=w_test_final)
        bal_acc = balanced_accuracy_score(y_test_final, y_pred_final, sample_weight=w_test_final)
    else:
        acc = accuracy_score(y_test_final, y_pred_final)
        f1 = f1_score(y_test_final, y_pred_final, average='weighted')
        bal_acc = balanced_accuracy_score(y_test_final, y_pred_final)

    # Weighted Confusion Matrix if use_weights and w_test_final is not None
    if use_weights and w_test_final is not None:
        df_conf = pd.DataFrame({
            'y_true': y_test_final,
            'y_pred': y_pred_final,
            'weight': w_test_final
        })
        weighted_cm = pd.pivot_table(
            df_conf, index='y_true', columns='y_pred',
            values='weight', aggfunc="sum", fill_value=0
        )
    else:
        cm = confusion_matrix(y_test_final, y_pred_final)
        weighted_cm = pd.DataFrame(
            cm,
            index=np.unique(y_test_final),
            columns=np.unique(y_test_final)
        )
    if label_map:
        weighted_cm = weighted_cm.rename(index=label_map, columns=label_map)

    cls_report = classification_report(
        y_test_final, y_pred_final,
        target_names=[label_map[i] for i in sorted(label_map)] if label_map else None,
        output_dict=True  # so we can store as structured data
    )

    # # Printing:
    # print("\n🎯 Accuracy:", acc)
    # print("⚖️ Balanced Accuracy:", bal_acc)
    # print("🧠 F1 Score (weighted):", f1)
    # print("\n📊 Confusion Matrix:")
    # print(confusion_matrix(y_test_final, y_pred_final))
    # print("\n📋 Classification Report:\n", classification_report(
    #     y_test_final, y_pred_final,
    #     target_names=[label_map[i] for i in sorted(label_map)] if label_map else None
    # ))
    # print("\n🧾 Weighted Confusion Matrix (named classes):")
    # print(weighted_cm)

    # Per-class metrics
    prec, rec, f1s, support = precision_recall_fscore_support(
        y_test_final, y_pred_final, average=None, sample_weight=w_test_final
    )
    labels = np.unique(y_test_final)

    # print("\n🔍 Per-Class Weighted Metrics:")
    # for i, label in enumerate(labels):
    #     class_name = label_map.get(label, str(label)) if label_map else str(label)
    #     print(f"Class '{class_name}' (label {label}): "
    #         f"Precision={prec[i]:.3f}, Recall={rec[i]:.3f}, "
    #         f"F1={f1s[i]:.3f}, Support={support[i]:.1f}")

    # Package Results
    results = {
        "model": best_model_pipeline,
        "y_pred": y_pred_final,
        "x_train": x_train_final,
        "y_train": y_train_final,
        "x_test": x_test_final,
        "y_test": y_test_final,
        "w_train": w_train_final,
        "w_test": w_test_final,
        "x_val": x_val_final,
        "y_val": y_val_final,
        "w_val": w_val_final,
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1": f1,
            "confusion_matrix": weighted_cm,
            "classification_report": cls_report
        }
    }

    return results

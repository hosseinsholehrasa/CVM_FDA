import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)

from predictive_modeling.core.data_preparation import finalize_dataset
from predictive_modeling.core.sampling import resample_dataset
from predictive_modeling.core.utils import find_best_threshold_for_death


def semi_supervised_training(
        model,
        X, Y,
        x_train, y_train, w_train,
        x_test_final, y_test_final, w_test,
        x_val, y_val, w_val,
        top_k_percent=0.6,
        resample_method="nosample",
        use_weights=False,
        num_passes=5,
        confidence_method="aum"  # "aum", "prob", "entropy"
    ):
    """
    Semi-supervised training with different confidence scoring methods.
    """

    # --- Collect unlabeled data ---
    X_nc = pd.concat([X[Y['Ongoing'] == 1], X[Y['Outcome Unknown'] == 1]])
    weights_nc = X_nc['number_of_animals'].astype(float)
    X_nc = X_nc.drop(columns=['number_of_animals'])

    # Ensure numeric conversions
    for col in ['animal_age', 'animal_weight', 'Topological Polar Surface Area']:
        if col in X_nc.columns:
            X_nc[col] = pd.to_numeric(X_nc[col], errors='coerce')
    X_nc = X_nc.dropna()

    # print("Unlabeled pool:", X_nc.shape)

    # --- Compute confidence scores ---
    if confidence_method == "aum":
        from collections import defaultdict

        # tracker: sample_idx -> list of margins across training passes
        aum_tracker = defaultdict(list)

        def compute_margins(model, X, y_true=None):
            """Compute margins from predict_proba outputs."""
            probs = model.predict_proba(X)
            if probs.shape[1] == 2:  # binary
                margin = np.abs(probs[:, 1] - probs[:, 0])
            else:  # multi-class
                sorted_probs = -np.sort(-probs, axis=1)  # sort descending
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            return margin

        # retrain multiple times (simulating epochs)
        for epoch in range(num_passes):
            model_epoch = clone(model)
            if use_weights and w_train is not None:
                model_epoch.fit(x_train, y_train, sample_weight=w_train)
            else:
                model_epoch.fit(x_train, y_train)

            margins = compute_margins(model_epoch, X_nc)

            # log margin per sample
            for i, m in enumerate(margins):
                aum_tracker[i].append(m)

        # final AUM score = mean margin over "epochs"
        scores = {i: np.mean(vals) for i, vals in aum_tracker.items()}

    elif confidence_method == "prob":
        probs = model.predict_proba(X_nc)
        scores = {i: max(p) for i, p in enumerate(probs)}

    elif confidence_method == "entropy":
        probs = model.predict_proba(X_nc)
        ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        scores = {i: -ent[i] for i in range(len(ent))}  # negative entropy = high confidence

    else:
        raise ValueError(f"Unknown confidence_method: {confidence_method}")

    # --- Select top-k confident samples ---
    score_df = pd.DataFrame.from_dict(scores, orient="index", columns=["score"]).reset_index()
    score_df = score_df.sort_values("score", ascending=False)
    k = int(len(score_df) * top_k_percent)
    selected_indices = score_df.iloc[:k]["index"].values

    X_confident = X_nc.iloc[selected_indices]
    weights_confident = weights_nc.iloc[selected_indices]
    pseudo_labels = model.predict(X_confident)

    # ---  Merge with labeled ---
    x_train_final_df = pd.DataFrame(x_train, columns=X_confident.columns)
    X_semi = pd.concat([x_train_final_df, X_confident], ignore_index=True)
    y_semi = np.concatenate([y_train, pseudo_labels])
    weights_semi = np.concatenate([w_train, weights_confident]) if use_weights else None

    # --- Optional resampling ---
    if resample_method != "nosample":

        x_train_sampled, y_train_sampled, w_train_sampled = resample_dataset(
            X_semi,
            pd.Series(y_semi),
            pd.Series(weights_semi) if weights_semi is not None else None,
            method=resample_method,
            use_weights=use_weights,
        )
        x_train_final, y_train_final, w_train_final = finalize_dataset(
            x_train_sampled,
            y_train_sampled,
            w_train_sampled,
        )
        x_test_final, y_test_final, w_test_final = x_test_final, y_test_final, w_test
        
    else:
        x_train_final, y_train_final = X_semi.to_numpy(), y_semi
        w_train_final = weights_semi
        x_test_final, y_test_final = x_test_final, y_test_final
        w_test_final = w_test

    # --- Retrain clone ---
    semi_model = clone(model)
    if use_weights and w_train_final is not None:
        semi_model.fit(x_train_final, y_train_final,
                       classifier__sample_weight=w_train_final)
    else:
        semi_model.fit(x_train_final, y_train_final)
    # --- Predict ---
    y_pred_final = semi_model.predict(x_test_final)
    # y_prob_final = semi_model.predict_proba(x_test_final)[:, 1]

    death_label = 0  # assuming 'Died' is label 0
    # --- Find best threshold on validation set ---
    best_thresh, best_f1 = find_best_threshold_for_death(semi_model, x_val, y_val, death_label=death_label)
    # best_thresh, best_f1 = find_best_threshold_overall(semi_model, x_val, y_val, average="weighted")

    # print(f"\n🔎 Best threshold from validation = {best_thresh:.2f}, F1 (Died={death_label}) = {best_f1:.3f}")

    # --- Evaluate on test set ---
    y_proba_test = semi_model.predict_proba(x_test_final)[:, death_label]
    y_pred_final = (y_proba_test >= best_thresh).astype(int) * death_label + (y_proba_test < best_thresh).astype(int) * (1 - death_label)

    # --- Evaluation  --- 
    if use_weights and w_test_final is not None:
        acc = accuracy_score(y_test_final, y_pred_final, sample_weight=w_test_final)
        bal_acc = balanced_accuracy_score(y_test_final, y_pred_final, sample_weight=w_test_final)
        f1 = f1_score(y_test_final, y_pred_final, average='weighted', sample_weight=w_test_final)
    else:
        acc = accuracy_score(y_test_final, y_pred_final)
        bal_acc = balanced_accuracy_score(y_test_final, y_pred_final)
        f1 = f1_score(y_test_final, y_pred_final, average='weighted')

    # # printing results
    # print("\n🎯 Accuracy:", acc)
    # print("⚖️ Balanced Accuracy:", bal_acc)
    # print("🧠 F1 Score (weighted):", f1)
    # print("\n📊 Confusion Matrix:")
    # print(confusion_matrix(y_test_final, y_pred_final))

    label_map = {0: "Died", 1: "Recovered/Normal"}
    cls_report = classification_report(
        y_test_final, y_pred_final,
        target_names=[label_map[i] for i in sorted(label_map)] if label_map else None
    )
    # print("\n📋 Classification Report:\n", cls_report)

    # Weighted Confusion Matrix + per-class metrics
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

        if label_map:
            weighted_cm_named = weighted_cm.rename(index=label_map, columns=label_map)
            # print("\n🧾 Weighted Confusion Matrix (named classes):")
            # print(weighted_cm_named)

        prec, rec, f1s, support = precision_recall_fscore_support(
            y_test_final, y_pred_final, average='weighted', sample_weight=w_test_final
        )
        labels = np.unique(y_test_final)
        # print("\n🔍 Per-Class Weighted Metrics:")
        # for i, label in enumerate(labels):
        #     class_name = label_map.get(label, str(label)) if label_map else str(label)
        #     print(f"Class '{class_name}' (label {label}): "
        #           f"Precision={prec[i]:.3f}, Recall={rec[i]:.3f}, "
        #           f"F1={f1s[i]:.3f}, Support={support[i]:.1f}")

    results = {
            "model": semi_model,
            "y_pred": y_pred_final,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test_final,
            "y_test": y_test_final,
            "x_semi": X_semi,
            "y_semi": y_semi,
            "w_train": None,
            "w_test": None,
            "metrics": {
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "f1": f1,
                # "confusion_matrix": weighted_cm,
                "classification_report": cls_report
            }
        }

    return results

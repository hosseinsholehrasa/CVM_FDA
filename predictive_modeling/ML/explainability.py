"""
All SHAP-based explainability utilities live here.
"""

import textwrap
from typing import Dict, Optional

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc_context


def fda_shap_summary_plot(
    model,
    shap_input: pd.DataFrame,
    cleaned_feature_names: Optional[list] = None,
    save_path: Optional[str] = None,
    max_display: int = 15,
    custom_rc: Optional[dict] = None,
):
    """
    Create and save a SHAP summary plot.

    Args:
        model: trained sklearn pipeline with named_steps['classifier']
        shap_input: feature dataframe
        cleaned_feature_names: optional list of pretty names
        save_path: path to save figure (PNG). If None → not saved.
        max_display: number of features to show
        custom_rc: matplotlib style overrides
    """

    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        explainer = shap.Explainer(model.named_steps["classifier"])
    else:
        explainer = shap.Explainer(model)

    shap_values = explainer(shap_input)

    if cleaned_feature_names is None:
        cleaned_feature_names = shap_input.columns.tolist()

    default_rc = {
        "font.size": 30,
        "font.weight": "bold",
        "axes.titlesize": 25,
        "axes.titleweight": "bold",
        "axes.labelsize": 24,
        "axes.labelweight": "bold",
        "xtick.labelsize": 23,
        "ytick.labelsize": 23,
    }

    rc = default_rc if custom_rc is None else custom_rc

    with rc_context(rc=rc):
        plt.figure(figsize=(12, 7))

        shap.summary_plot(
            shap_values,
            shap_input,
            feature_names=cleaned_feature_names,
            show=False,
            max_display=max_display,
        )

        plt.xlabel("Mean SHAP Value (← Death | Recovery →)", weight="bold")

        # Remove top/right spines
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        if save_path:
            plt.savefig(save_path, dpi=1200, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        plt.show()
        plt.close()

    return shap_values


def fda_plot_top_bottom_shap(
    shap_values,
    x_test: pd.DataFrame,
    field_name: str,
    label_encoders: Dict,
    min_samples: int = 30,
    top_n: int = 10,
    y_axis_label: Optional[str] = None,
    save_path: Optional[str] = None,
    custom_rc: Optional[dict] = None,
):
    """
    Bar plot for Top & Bottom mean SHAP values.

    Returns:
        pd.DataFrame: table with aggregated SHAP values used for the plot
    """

    # Convert SHAP → DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=x_test.columns)

    combined = pd.DataFrame({
        field_name: x_test[field_name].values,
        "shap_value": shap_df[field_name].values,
    })

    # Decode labels if encoded
    if field_name in label_encoders:
        le = label_encoders[field_name]
        combined[field_name] = le.inverse_transform(
            combined[field_name].astype(int)
        )

    # Aggregate
    grouped = (
        combined
        .groupby(field_name)
        .agg(mean_shap=("shap_value", "mean"),
             count=("shap_value", "size"))
        .reset_index()
    )

    grouped = grouped[grouped["count"] >= min_samples]
    grouped_sorted = grouped.sort_values("mean_shap")

    # Top & bottom
    bottom_n = grouped_sorted.head(top_n)
    top_n_df = grouped_sorted.tail(top_n)
    final = pd.concat([bottom_n, top_n_df])

    # Clean very long drug names
    if field_name == "name":
        final[field_name] = (
            final[field_name]
            .str.replace(
                "(6R,25R)-5-O-Demethyl-28-deoxy-6,28-epoxy-25-methylmilbemycin B",
                "Milbemycin A3",
                regex=False,
            )
            .str.replace("(USAN:USP:INN:BAN)", "", regex=False)
        )

    # Wrap long labels nicely
    final[field_name] = final[field_name].apply(
        lambda x: "\n".join(textwrap.wrap(x, 45))
    )

    # Color scheme
    colors = final["mean_shap"].apply(
        lambda x: "#2ecc71" if x > 0 else "#e74c3c"
    )

    default_rc = {
        "font.size": 30,
        "font.weight": "bold",
        "axes.titlesize": 25,
        "axes.titleweight": "bold",
        "axes.labelsize": 20,
        "axes.labelweight": "bold",
        "xtick.labelsize": 23,
        "ytick.labelsize": 23,
    }

    rc = default_rc if custom_rc is None else custom_rc
    with rc_context(rc=rc):
        fig, ax = plt.subplots(figsize=(14, 12))

        bars = ax.barh(
            final[field_name],
            final["mean_shap"],
            color=colors,
            edgecolor="black",
            linewidth=1.2,
            height=0.6,
        )

        # Shadow effect
        for bar, val in zip(bars, final["mean_shap"]):
            bar.set_zorder(2)

            if val >= 0:
                shadow_x = bar.get_x() + 0.05
                shadow_width = bar.get_width()
            else:
                shadow_x = bar.get_x() - 0.05 + bar.get_width()
                shadow_width = abs(bar.get_width())

            ax.add_patch(
                patches.Rectangle(
                    (shadow_x, bar.get_y() - 0.05),
                    shadow_width,
                    bar.get_height(),
                    linewidth=0,
                    facecolor="grey",
                    alpha=0.2,
                    zorder=1,
                )
            )

        ax.axvline(0, color="black", linewidth=1.5)

        ax.set_xlabel(
            "Mean SHAP Value (← Death | Recovery →)",
            weight="bold",
        )

        ax.set_ylabel(
            y_axis_label if y_axis_label else field_name,
            weight="bold",
        )

        ax.set_title(
            "Top & Bottom Mean SHAP Values",
            weight="bold",
            pad=10,
        )

        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}")
        )

        # Clean spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # plt.subplots_adjust(left=0.55)  # increases left margin for y-axis space
        if save_path:
            plt.savefig(save_path, dpi=1200, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        plt.tight_layout()
        plt.show()
        plt.close()

    return final

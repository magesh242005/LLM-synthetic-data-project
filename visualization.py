"""
visualization.py
================
Module for producing publication-ready, aesthetic charts comparing Original and Synthetic datasets,
model performance, confusion matrices, ROC curves, feature importances, and SHAP explainability.
Automatically exports high-resolution visual assets to the visuals/ directory.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def ensure_visuals_dir(directory: str = "visuals") -> str:
    os.makedirs(directory, exist_ok=True)
    return directory


def plot_histograms_comparison(
    df_orig: pd.DataFrame,
    df_synth: pd.DataFrame,
    numerical_cols: List[str],
    save_path: str = "visuals/distribution_comparison_histograms.png"
) -> str:
    """
    Generate side-by-side or overlaid distribution histograms & KDE curves.
    """
    ensure_visuals_dir(os.path.dirname(save_path) or "visuals")
    cols = [c for c in numerical_cols if c in df_orig.columns and c in df_synth.columns][:6]
    if not cols:
        return ""

    n_cols = len(cols)
    fig, axes = plt.subplots(nrows=n_cols, ncols=1, figsize=(10, 3.2 * n_cols), sharex=False)
    if n_cols == 1:
        axes = [axes]

    for idx, col in enumerate(cols):
        ax = axes[idx]
        s_orig = pd.to_numeric(df_orig[col], errors='coerce').dropna()
        s_synth = pd.to_numeric(df_synth[col], errors='coerce').dropna()

        sns.kdeplot(s_orig, ax=ax, label="Original Data", color="#1f77b4", fill=True, alpha=0.35, linewidth=2)
        sns.kdeplot(s_synth, ax=ax, label="Synthetic Data", color="#ff7f0e", fill=True, alpha=0.35, linewidth=2, linestyle="--")

        ax.set_title(f"Distribution Comparison: {col}", fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved histogram comparison to {save_path}")
    return save_path


def plot_boxplots_comparison(
    df_orig: pd.DataFrame,
    df_synth: pd.DataFrame,
    numerical_cols: List[str],
    save_path: str = "visuals/boxplots_comparison.png"
) -> str:
    """
    Generate paired boxplots comparing spread, median, and outliers.
    """
    ensure_visuals_dir(os.path.dirname(save_path) or "visuals")
    cols = [c for c in numerical_cols if c in df_orig.columns and c in df_synth.columns][:6]
    if not cols:
        return ""

    df_o = df_orig[cols].copy()
    df_o['Dataset_Type'] = 'Original'

    df_s = df_synth[cols].copy()
    df_s['Dataset_Type'] = 'Synthetic'

    combined = pd.concat([df_o, df_s], ignore_index=True)
    melted = pd.melt(combined, id_vars=['Dataset_Type'], value_vars=cols, var_name='Feature', value_name='Value')

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=melted,
        x='Feature',
        y='Value',
        hue='Dataset_Type',
        palette={'Original': '#2b5c8f', 'Synthetic': '#e26d5c'},
        fliersize=3
    )
    plt.title("Feature Spread & Quantile Comparison (Boxplots)", fontsize=14, fontweight='bold', pad=12)
    plt.xticks(rotation=25, ha='right', fontsize=10)
    plt.xlabel("Features", fontsize=11)
    plt.ylabel("Standardized / Raw Value", fontsize=11)
    plt.legend(title="Dataset", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved boxplot comparison to {save_path}")
    return save_path


def plot_correlation_heatmaps(
    df_orig: pd.DataFrame,
    df_synth: pd.DataFrame,
    numerical_cols: List[str],
    save_path: str = "visuals/correlation_heatmaps.png"
) -> str:
    """
    Plot 3-panel correlation comparison: Original, Synthetic, and Absolute Difference.
    """
    ensure_visuals_dir(os.path.dirname(save_path) or "visuals")
    cols = [c for c in numerical_cols if c in df_orig.columns and c in df_synth.columns]
    if len(cols) < 2:
        return ""

    corr_orig = df_orig[cols].corr()
    corr_synth = df_synth[cols].corr()
    corr_diff = (corr_orig - corr_synth).abs()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(corr_orig, ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
    axes[0].set_title("Original Dataset Correlation", fontsize=12, fontweight='bold')

    sns.heatmap(corr_synth, ax=axes[1], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
    axes[1].set_title("Synthetic Dataset Correlation", fontsize=12, fontweight='bold')

    sns.heatmap(corr_diff, ax=axes[2], annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, cbar=True)
    axes[2].set_title("Correlation Absolute Difference (|Orig - Synth|)", fontsize=12, fontweight='bold')

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correlation heatmaps to {save_path}")
    return save_path


def plot_class_balance(
    df_orig: pd.DataFrame,
    df_synth: pd.DataFrame,
    target_col: str,
    save_path: str = "visuals/class_balance_comparison.png"
) -> str:
    """
    Plot side-by-side bar chart of target class distributions.
    """
    ensure_visuals_dir(os.path.dirname(save_path) or "visuals")
    if target_col not in df_orig.columns or target_col not in df_synth.columns:
        return ""

    vc_orig = df_orig[target_col].value_counts(normalize=True).reset_index()
    vc_orig.columns = [target_col, 'Proportion']
    vc_orig['Dataset'] = 'Original'

    vc_synth = df_synth[target_col].value_counts(normalize=True).reset_index()
    vc_synth.columns = [target_col, 'Proportion']
    vc_synth['Dataset'] = 'Synthetic'

    combined = pd.concat([vc_orig, vc_synth], ignore_index=True)
    combined[target_col] = combined[target_col].astype(str)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=combined,
        x=target_col,
        y='Proportion',
        hue='Dataset',
        palette={'Original': '#1f77b4', 'Synthetic': '#ff7f0e'}
    )
    plt.title(f"Target Class Balance: {target_col} (Original vs Synthetic)", fontsize=13, fontweight='bold', pad=10)
    plt.ylabel("Relative Frequency / Proportion", fontsize=11)
    plt.xlabel(f"Class / Label ({target_col})", fontsize=11)
    plt.ylim(0, 1.05)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1%}", (p.get_x() + p.get_width() / 2., height + 0.02),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved class balance chart to {save_path}")
    return save_path


def plot_categorical_distributions(
    df_orig: pd.DataFrame,
    df_synth: pd.DataFrame,
    categorical_cols: List[str],
    save_path: str = "visuals/categorical_distributions.png"
) -> str:
    """
    Plot bar comparisons of categorical proportions.
    """
    ensure_visuals_dir(os.path.dirname(save_path) or "visuals")
    cols = [c for c in categorical_cols if c in df_orig.columns and c in df_synth.columns][:4]
    if not cols:
        return ""

    n_cols = len(cols)
    fig, axes = plt.subplots(nrows=n_cols, ncols=1, figsize=(10, 3.5 * n_cols))
    if n_cols == 1:
        axes = [axes]

    for idx, col in enumerate(cols):
        ax = axes[idx]
        vc_o = df_orig[col].astype(str).value_counts(normalize=True).reset_index()
        vc_o.columns = ['Category', 'Proportion']
        vc_o['Dataset'] = 'Original'

        vc_s = df_synth[col].astype(str).value_counts(normalize=True).reset_index()
        vc_s.columns = ['Category', 'Proportion']
        vc_s['Dataset'] = 'Synthetic'

        comb = pd.concat([vc_o, vc_s], ignore_index=True)

        sns.barplot(data=comb, x='Category', y='Proportion', hue='Dataset', ax=ax, palette={'Original': '#3470a3', 'Synthetic': '#e87461'})
        ax.set_title(f"Categorical Profile: {col}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Proportion", fontsize=10)
        ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved categorical distributions to {save_path}")
    return save_path


def plot_confusion_matrices(
    cm_dict: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    save_path: str = "visuals/confusion_matrix.png"
) -> str:
    """
    Plot a grid of confusion matrices for all evaluated models across regimes.
    """
    ensure_visuals_dir(os.path.dirname(save_path) or "visuals")
    num_cms = len(cm_dict)
    if num_cms == 0:
        return ""

    ncols = min(3, num_cms)
    nrows = int(np.ceil(num_cms / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for idx, (title, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=labels or [0, 1], yticklabels=labels or [0, 1])
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    # Hide extra unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix grid to {save_path}")
    return save_path


def plot_model_comparison_bars(
    results_df: pd.DataFrame,
    save_path: str = "visuals/model_performance_comparison.png"
) -> str:
    """
    Bar plot comparing Accuracy, F1, and ROC-AUC across models and regimes.
    """
    ensure_visuals_dir(os.path.dirname(save_path) or "visuals")
    if results_df.empty:
        return ""

    plt.figure(figsize=(12, 6))
    melted = pd.melt(
        results_df,
        id_vars=['Model', 'Regime'],
        value_vars=[col for col in ['Accuracy', 'F1 Score', 'ROC-AUC'] if col in results_df.columns],
        var_name='Metric',
        value_name='Score'
    )

    sns.barplot(
        data=melted,
        x='Model',
        y='Score',
        hue='Regime',
        palette={'Original Only': '#1f77b4', 'Synthetic Only (TSTR)': '#ff7f0e', 'Combined (Augmented)': '#2ca02c'}
    )
    plt.title("Model Performance Across Training Regimes (Tested on Real Holdout Set)", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0, 1.1)
    plt.ylabel("Evaluation Score", fontsize=11)
    plt.xlabel("Machine Learning Algorithm", fontsize=11)
    plt.legend(title="Training Regime", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model comparison bars to {save_path}")
    return save_path


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: str = "visuals/feature_importance.png"
) -> str:
    """
    Plot feature importance ranking comparison between Original-trained and Synthetic-trained models.
    """
    ensure_visuals_dir(os.path.dirname(save_path) or "visuals")
    if importance_df.empty:
        return ""

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x='Importance',
        y='Feature',
        hue='Trained_On',
        palette={'Original Data': '#1f77b4', 'Synthetic Data': '#ff7f0e'}
    )
    plt.title("Feature Importance Alignment (Original vs Synthetic Model)", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Normalized Gini / Split Importance", fontsize=11)
    plt.ylabel("Features", fontsize=11)
    plt.legend(title="Training Dataset", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance chart to {save_path}")
    return save_path


def plot_privacy_distance(
    distances: np.ndarray,
    save_path: str = "visuals/privacy_dcr_distribution.png"
) -> str:
    """
    Plot Distance to Closest Record (DCR) distribution for privacy validation.
    """
    ensure_visuals_dir(os.path.dirname(save_path) or "visuals")
    plt.figure(figsize=(8, 4.5))
    sns.histplot(distances, kde=True, color="#9467bd", bins=25, edgecolor="black", alpha=0.6)
    median_d = float(np.median(distances))
    pct_5th = float(np.percentile(distances, 5))

    plt.axvline(median_d, color='red', linestyle='--', linewidth=2, label=f'Median DCR: {median_d:.3f}')
    plt.axvline(pct_5th, color='orange', linestyle=':', linewidth=2, label=f'5th Percentile DCR: {pct_5th:.3f}')

    plt.title("Distance to Closest Real Record (DCR) Privacy Profile", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Euclidean Distance to Nearest Real Record", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved privacy DCR chart to {save_path}")
    return save_path

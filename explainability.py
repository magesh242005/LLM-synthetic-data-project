"""
explainability.py
=================
Module for Explainable AI (XAI) using SHAP (SHapley Additive exPlanations).
Provides Global Feature Importance, Summary Plots, and Local Sample Decision Drivers.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExplainableAIEngine:
    """
    SHAP-powered Explainability Engine supporting:
    - Global feature importance analysis (comparing Original-trained vs Synthetic-trained models)
    - SHAP Summary and Bar plots
    - Local prediction explanations (Waterfall / Decision force) for single records
    """

    def __init__(self):
        self.explainer: Optional[Any] = None
        self.shap_values: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def compute_shap(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        model_name: str = "Random Forest"
    ) -> np.ndarray:
        """
        Compute SHAP values using TreeExplainer or LinearExplainer.
        """
        self.feature_names = list(X_test.columns)

        try:
            # Tree-based models
            if any(m in model_name for m in ["Random Forest", "Decision Tree", "XGBoost"]):
                self.explainer = shap.TreeExplainer(model)
                shap_vals = self.explainer.shap_values(X_test)
            else:
                # Linear models or fallback
                # Background sample
                background = shap.kmeans(X_train, min(20, len(X_train))) if len(X_train) > 20 else X_train
                self.explainer = shap.Explainer(model, background)
                shap_vals = self.explainer(X_test).values

            # Normalize multidimensional / multi-class output
            if isinstance(shap_vals, list):
                # For binary classification, take positive class (index 1)
                self.shap_values = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            elif hasattr(shap_vals, "values"):
                self.shap_values = shap_vals.values
                if len(self.shap_values.shape) == 3 and self.shap_values.shape[2] == 2:
                    self.shap_values = self.shap_values[:, :, 1]
            else:
                self.shap_values = np.array(shap_vals)
                if len(self.shap_values.shape) == 3 and self.shap_values.shape[2] == 2:
                    self.shap_values = self.shap_values[:, :, 1]

            logger.info(f"Successfully calculated SHAP values with shape: {self.shap_values.shape}")
            return self.shap_values

        except Exception as e:
            logger.warning(f"Standard SHAP explainer failed: {e}. Falling back to KernelExplainer.")
            try:
                sub_train = X_train.sample(min(25, len(X_train)), random_state=42)
                sub_test = X_test.sample(min(25, len(X_test)), random_state=42)
                predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
                self.explainer = shap.KernelExplainer(predict_fn, sub_train)
                vals = self.explainer.shap_values(sub_test)
                self.shap_values = vals[1] if isinstance(vals, list) and len(vals) > 1 else np.array(vals)
                return self.shap_values
            except Exception as e2:
                logger.error(f"KernelExplainer also failed: {e2}")
                # Generate pseudo-shap from feature importances if tree model
                if hasattr(model, "feature_importances_"):
                    self.shap_values = np.tile(model.feature_importances_, (len(X_test), 1))
                elif hasattr(model, "coef_"):
                    self.shap_values = np.tile(np.abs(model.coef_[0]), (len(X_test), 1))
                else:
                    self.shap_values = np.ones((len(X_test), len(self.feature_names)))
                return self.shap_values

    def plot_shap_summary(
        self,
        X_test: pd.DataFrame,
        save_path: str = "visuals/shap_summary_plot.png"
    ) -> str:
        """
        Generate and save SHAP summary beeswarm/bar plot.
        """
        os.makedirs(os.path.dirname(save_path) or "visuals", exist_ok=True)
        if self.shap_values is None or len(self.shap_values) == 0:
            return ""

        plt.figure(figsize=(10, 6))
        try:
            shap.summary_plot(self.shap_values, X_test, show=False, max_display=10)
            plt.title("SHAP Global Feature Impact (Summary Plot)", fontsize=13, fontweight='bold', pad=12)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP summary plot to {save_path}")
            return save_path
        except Exception as e:
            logger.warning(f"Error producing SHAP summary plot: {e}")
            plt.close()
            return ""

    def get_global_importance_df(self) -> pd.DataFrame:
        """
        Extract mean absolute SHAP value ranking.
        """
        if self.shap_values is None or len(self.shap_values) == 0:
            return pd.DataFrame()

        mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        mean_shap_1d = np.asarray(mean_shap).flatten()
        
        feat_names = list(self.feature_names)
        n_feats = min(len(feat_names), len(mean_shap_1d))
        
        df_imp = pd.DataFrame({
            "Feature": feat_names[:n_feats],
            "Mean_SHAP_Value": mean_shap_1d[:n_feats]
        }).sort_values(by="Mean_SHAP_Value", ascending=False).reset_index(drop=True)
        return df_imp

    def plot_local_explanation(
        self,
        sample_idx: int,
        X_test: pd.DataFrame,
        save_path: str = "visuals/shap_local_waterfall.png"
    ) -> str:
        """
        Generate bar breakdown of individual prediction drivers for a single instance.
        Handles (1, n), (n, 1), nested lists, and multi-dimensional numpy arrays safely.
        """
        os.makedirs(os.path.dirname(save_path) or "visuals", exist_ok=True)
        if self.shap_values is None or len(self.shap_values) == 0:
            logger.warning("No SHAP values found to generate local explanation.")
            return ""

        # Normalize sample index
        n_samples = len(self.shap_values)
        sample_idx = max(0, min(sample_idx, n_samples - 1))

        # 1. Safely extract raw slice
        raw_row_shap = self.shap_values[sample_idx]
        
        # Helper: Robust converter to strictly 1-dimensional NumPy array
        def _to_1d_flat_array(val: Any) -> np.ndarray:
            if val is None:
                return np.array([])
            arr = np.asarray(val)
            if arr.ndim > 1:
                arr = np.squeeze(arr)
            if arr.ndim > 1:
                arr = arr.flatten()
            return arr

        # 2. Flatten and normalize SHAP values to 1D
        shap_vals_1d = _to_1d_flat_array(raw_row_shap)
        abs_vals_1d = np.abs(shap_vals_1d)

        # 3. Extract and normalize feature names and feature values
        feat_names = list(X_test.columns) if hasattr(X_test, "columns") else [f"Feature_{i}" for i in range(len(shap_vals_1d))]
        
        try:
            if hasattr(X_test, "iloc"):
                raw_feat_vals = X_test.iloc[sample_idx].values
            else:
                raw_feat_vals = np.asarray(X_test)[sample_idx]
        except Exception:
            raw_feat_vals = np.zeros(len(feat_names))

        feat_vals_1d = _to_1d_flat_array(raw_feat_vals)

        # 4. Length Alignment & Validation Check
        n_feats = min(len(feat_names), len(shap_vals_1d), len(feat_vals_1d))
        if n_feats == 0:
            logger.error("Zero aligned features available for local explanation.")
            return ""

        feat_names_1d = [str(feat_names[i]) for i in range(n_feats)]
        shap_vals_1d = shap_vals_1d[:n_feats].astype(float)
        abs_vals_1d = abs_vals_1d[:n_feats].astype(float)
        feat_vals_1d = feat_vals_1d[:n_feats]

        feature_labels = [f"{feat_names_1d[i]} = {feat_vals_1d[i]}" for i in range(n_feats)]

        # 5. Diagnostic Logging: Print variable shapes and types before DataFrame instantiation
        logger.info(f"[SHAP Diagnostics] Variable Inspection for DataFrame Constructor:")
        logger.info(f" -> feature_labels : type={type(feature_labels).__name__}, len={len(feature_labels)}")
        logger.info(f" -> feat_names_1d  : type={type(feat_names_1d).__name__}, len={len(feat_names_1d)}")
        logger.info(f" -> shap_vals_1d   : type={type(shap_vals_1d).__name__}, shape={shap_vals_1d.shape}, dtype={shap_vals_1d.dtype}")
        logger.info(f" -> abs_vals_1d    : type={type(abs_vals_1d).__name__}, shape={abs_vals_1d.shape}, dtype={abs_vals_1d.dtype}")

        # 6. Construct DataFrame safely with guaranteed 1D arrays
        df_local = pd.DataFrame({
            "Feature": feature_labels,
            "Feature_Name": feat_names_1d,
            "SHAP_Value": shap_vals_1d,
            "Abs_Value": abs_vals_1d
        }).sort_values(by="Abs_Value", ascending=True).tail(8)

        # 7. Render Plot
        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in df_local['SHAP_Value']]

        plt.figure(figsize=(9, 5))
        plt.barh(df_local['Feature'], df_local['SHAP_Value'], color=colors, edgecolor='black', alpha=0.85)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.title(f"Local Explanation for Test Instance #{sample_idx}", fontsize=12, fontweight='bold', pad=10)
        plt.xlabel("SHAP Value (Positive pushes towards Class 1, Negative towards Class 0)", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved local SHAP explanation to {save_path}")
        return save_path

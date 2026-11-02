"""
model_training.py
=================
Module for automated machine learning model training across 3 regimes:
1. Model trained on Original Dataset
2. Model trained on Synthetic Dataset (TSTR - Train on Synthetic, Test on Real)
3. Model trained on Combined Dataset (Augmented)

Models supported: Logistic Regression, Decision Tree, Random Forest, XGBoost.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    # Test if XGBClassifier can be initialized
    _test_xgb = xgb.XGBClassifier(n_estimators=2, max_depth=2)
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from evaluation import evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Automated Machine Learning Pipeline that orchestrates model training across
    all three experimental regimes and tests on holdout real-world test sets.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.trained_models: Dict[str, Any] = {}
        self.evaluation_results: List[Dict[str, Any]] = []
        self.predictions: Dict[str, Dict[str, np.ndarray]] = {}

    def _get_base_models(self) -> Dict[str, Any]:
        """
        Instantiate standard classifiers.
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=self.random_state)
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                eval_metric="logloss",
                random_state=self.random_state
            )
        else:
            models["XGBoost"] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=self.random_state
            )

        return models

    def train_and_evaluate_regime(
        self,
        regime_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test_real: pd.DataFrame,
        y_test_real: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        Train all models under a specific regime and evaluate against real holdout test set.
        """
        logger.info(f"Training regime '{regime_name}' with {len(X_train)} training records...")
        base_models = self._get_base_models()
        regime_results = []

        for model_name, model in base_models.items():
            model_key = f"{model_name} ({regime_name})"
            try:
                # Fit model
                model.fit(X_train, y_train)
                self.trained_models[model_key] = model

                # Save model to disk
                save_filename = os.path.join(self.models_dir, f"{model_name.lower().replace(' ', '_')}_{regime_name.lower().replace(' ', '_')}.joblib")
                joblib.dump(model, save_filename)

                # Predictions on REAL test set
                y_pred = model.predict(X_test_real)
                y_prob = model.predict_proba(X_test_real) if hasattr(model, "predict_proba") else None

                self.predictions[model_key] = {
                    "y_pred": y_pred,
                    "y_prob": y_prob
                }

                # Evaluate metrics
                metrics = evaluate_classifier(y_test_real, y_pred, y_prob)

                result_entry = {
                    "Model": model_name,
                    "Regime": regime_name,
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1 Score": metrics["F1 Score"],
                    "ROC-AUC": metrics["ROC-AUC"],
                    "Confusion_Matrix": metrics["Confusion_Matrix"],
                    "Classification_Report_Text": metrics["Classification_Report_Text"],
                    "Classification_Report_Dict": metrics["Classification_Report_Dict"]
                }
                regime_results.append(result_entry)
                self.evaluation_results.append(result_entry)

                logger.info(f"[{model_key}] Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1 Score']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f}")

            except Exception as e:
                logger.error(f"Error training {model_key}: {e}")

        return regime_results

    def run_full_pipeline(
        self,
        X_train_orig: pd.DataFrame,
        y_train_orig: pd.Series,
        X_test_real: pd.DataFrame,
        y_test_real: pd.Series,
        X_synth: pd.DataFrame,
        y_synth: pd.Series
    ) -> pd.DataFrame:
        """
        Execute full comparative pipeline across all 3 regimes:
        1. Original Only
        2. Synthetic Only (TSTR)
        3. Combined (Original + Synthetic)
        """
        self.evaluation_results.clear()
        self.predictions.clear()

        # Regime 1: Original Dataset
        self.train_and_evaluate_regime(
            regime_name="Original Only",
            X_train=X_train_orig,
            y_train=y_train_orig,
            X_test_real=X_test_real,
            y_test_real=y_test_real
        )

        # Regime 2: Synthetic Dataset (TSTR)
        self.train_and_evaluate_regime(
            regime_name="Synthetic Only (TSTR)",
            X_train=X_synth,
            y_train=y_synth,
            X_test_real=X_test_real,
            y_test_real=y_test_real
        )

        # Regime 3: Combined (Augmented)
        X_comb = pd.concat([X_train_orig, X_synth], ignore_index=True)
        y_comb = pd.concat([y_train_orig, y_synth], ignore_index=True)

        self.train_and_evaluate_regime(
            regime_name="Combined (Augmented)",
            X_train=X_comb,
            y_train=y_comb,
            X_test_real=X_test_real,
            y_test_real=y_test_real
        )

        df_results = pd.DataFrame(self.evaluation_results)
        return df_results

    def get_feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importances from Random Forest / XGBoost models across Original and Synthetic regimes.
        """
        records = []
        for regime, label in [("Original Only", "Original Data"), ("Synthetic Only (TSTR)", "Synthetic Data")]:
            rf_key = f"Random Forest ({regime})"
            if rf_key in self.trained_models:
                rf = self.trained_models[rf_key]
                if hasattr(rf, "feature_importances_"):
                    for feat, imp in zip(feature_names, rf.feature_importances_):
                        records.append({
                            "Feature": feat,
                            "Importance": float(imp),
                            "Trained_On": label
                        })

        return pd.DataFrame(records)

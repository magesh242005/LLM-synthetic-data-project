"""
preprocessing.py
================
Module for dataset loading, automated schema detection, missing value imputation,
duplicate removal, categorical encoding, feature scaling, and statistical profiling.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    """
    Comprehensive Data Preprocessor handling:
    - Automatic column type detection (Numerical vs Categorical vs Identifier/Datetime)
    - Missing value imputation and duplicate removal
    - Statistical profiling
    - Categorical label encoding and feature scaling
    - Train / Test stratified splitting
    """

    def __init__(self):
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.target_col: Optional[str] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.column_meta: Dict[str, Any] = {}

    def load_data(self, file_source: Any, max_rows: int = 5000) -> pd.DataFrame:
        """
        Load dataset from file path, StringIO, or UploadedFile with smart sampling for large datasets.
        """
        try:
            if isinstance(file_source, str):
                if not os.path.exists(file_source):
                    raise FileNotFoundError(f"File not found: {file_source}")
                df = pd.read_csv(file_source)
            else:
                df = pd.read_csv(file_source)

            # Cap large datasets to max_rows for lightning-fast UI responsiveness
            if len(df) > max_rows:
                logger.info(f"Dataset has {len(df)} rows. Sampling {max_rows} rows for high-performance interactive processing.")
                df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

            self.raw_df = df.copy()
            logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def detect_column_types(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        Automatically detect numerical and categorical columns.
        """
        self.numerical_cols = []
        self.categorical_cols = []
        self.target_col = target_col

        for col in df.columns:
            if target_col and col == target_col:
                continue

            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # If unique values are very low (<= 4) and integer, could be categorical flag, but keep numeric if float
                unique_vals = df[col].nunique(dropna=True)
                if unique_vals <= 4 and df[col].dtype in ['int64', 'int32', 'bool', 'object']:
                    self.categorical_cols.append(col)
                else:
                    self.numerical_cols.append(col)
            else:
                self.categorical_cols.append(col)

        logger.info(f"Detected {len(self.numerical_cols)} numerical cols: {self.numerical_cols}")
        logger.info(f"Detected {len(self.categorical_cols)} categorical cols: {self.categorical_cols}")
        return self.numerical_cols, self.categorical_cols

    def clean_data(
        self,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        handle_missing: bool = True,
        missing_strategy_num: str = "median",
        missing_strategy_cat: str = "mode"
    ) -> pd.DataFrame:
        """
        Clean raw data: deduplicate and impute missing values.
        """
        df_clean = df.copy()

        # Remove duplicate rows
        if drop_duplicates:
            init_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            dropped = init_rows - len(df_clean)
            if dropped > 0:
                logger.info(f"Removed {dropped} duplicate rows.")

        # Handle missing values
        if handle_missing:
            # Numerical imputation
            for col in self.numerical_cols:
                if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                    if missing_strategy_num == "median":
                        fill_val = df_clean[col].median()
                    elif missing_strategy_num == "mean":
                        fill_val = df_clean[col].mean()
                    else:
                        fill_val = 0
                    df_clean[col] = df_clean[col].fillna(fill_val)

            # Categorical imputation
            for col in self.categorical_cols:
                if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                    if missing_strategy_cat == "mode":
                        mode_series = df_clean[col].mode()
                        fill_val = mode_series[0] if not mode_series.empty else "Missing"
                    else:
                        fill_val = "Unknown"
                    df_clean[col] = df_clean[col].fillna(fill_val)

            # Target column imputation if target exists
            if self.target_col and self.target_col in df_clean.columns:
                if df_clean[self.target_col].isnull().sum() > 0:
                    df_clean = df_clean.dropna(subset=[self.target_col]).reset_index(drop=True)

        self.cleaned_df = df_clean
        return df_clean

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute descriptive statistical profile for numerical and categorical features.
        """
        summary: Dict[str, Any] = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "missing_cells": int(df.isnull().sum().sum()),
            "missing_percentage": float(np.round((df.isnull().sum().sum() / (df.size or 1)) * 100, 2)),
            "duplicate_records": int(df.duplicated().sum()),
            "numerical_summary": {},
            "categorical_summary": {}
        }

        # Numerical columns profile
        for col in self.numerical_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if not series.empty:
                    summary["numerical_summary"][col] = {
                        "count": int(series.count()),
                        "mean": float(np.round(series.mean(), 4)),
                        "std": float(np.round(series.std(), 4)),
                        "median": float(np.round(series.median(), 4)),
                        "min": float(np.round(series.min(), 4)),
                        "max": float(np.round(series.max(), 4)),
                        "q25": float(np.round(series.quantile(0.25), 4)),
                        "q75": float(np.round(series.quantile(0.75), 4)),
                        "skewness": float(np.round(series.skew(), 4)) if len(series) > 2 else 0.0
                    }

        # Categorical columns profile
        for col in self.categorical_cols:
            if col in df.columns:
                val_counts = df[col].astype(str).value_counts()
                top_cats = val_counts.head(5).to_dict()
                summary["categorical_summary"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_categories": top_cats,
                    "mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else "N/A"
                }

        # Target profile if provided
        if self.target_col and self.target_col in df.columns:
            target_counts = df[self.target_col].value_counts(normalize=True).to_dict()
            summary["target_distribution"] = {str(k): float(np.round(v, 4)) for k, v in target_counts.items()}

        return summary

    def encode_and_scale(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Encode categorical features using LabelEncoder and scale numerical features.
        Returns transformed X (DataFrame) and y (Series or None).
        """
        df_encoded = df.copy()
        target_series = None

        if target_col and target_col in df_encoded.columns:
            target_raw = df_encoded[target_col]
            # If target is categorical, encode to integer
            if not pd.api.types.is_numeric_dtype(target_raw):
                if fit or (target_col not in self.label_encoders):
                    le_target = LabelEncoder()
                    target_series = pd.Series(le_target.fit_transform(target_raw.astype(str)), name=target_col)
                    self.label_encoders[target_col] = le_target
                else:
                    target_series = pd.Series(self.label_encoders[target_col].transform(target_raw.astype(str)), name=target_col)
            else:
                target_series = target_raw.copy()
            df_encoded = df_encoded.drop(columns=[target_col])

        # Encode categorical columns
        for col in self.categorical_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(str)
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen labels gracefully
                        classes = set(le.classes_)
                        df_encoded[col] = df_encoded[col].apply(lambda x: x if x in classes else le.classes_[0])
                        df_encoded[col] = le.transform(df_encoded[col])
                    else:
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df_encoded[col])
                        self.label_encoders[col] = le

        # Ensure all columns in X are numeric
        for col in df_encoded.columns:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

        # Scale numerical features with StandardScaler
        if self.numerical_cols:
            valid_num_cols = [c for c in self.numerical_cols if c in df_encoded.columns]
            if valid_num_cols:
                if fit or (self.scaler is None):
                    self.scaler = StandardScaler()
                    df_encoded[valid_num_cols] = self.scaler.fit_transform(df_encoded[valid_num_cols])
                else:
                    df_encoded[valid_num_cols] = self.scaler.transform(df_encoded[valid_num_cols])

        return df_encoded, target_series

    def split_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.25,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Stratified train/test split.
        """
        # Determine if stratification is applicable
        stratify = y if y.nunique() <= 10 and (y.value_counts().min() >= 2) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        logger.info(f"Split dataset: Train size = {len(X_train)}, Test size = {len(X_test)}")
        return X_train, X_test, y_train, y_test

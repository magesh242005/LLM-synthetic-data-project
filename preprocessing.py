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


def ensure_sample_datasets(base_dir: Optional[str] = None) -> None:
    """
    Ensure all default sample datasets exist in the data/ directory.
    If missing (e.g. on cloud platforms like Streamlit Cloud), auto-generate them.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    sample_finance = os.path.join(data_dir, "sample_dataset.csv")
    if not os.path.exists(sample_finance) or os.path.getsize(sample_finance) == 0:
        content = """Age,Gender,Annual_Income,Credit_Score,Tenure_Months,Account_Balance,Num_Products,Has_Credit_Card,Is_Active_Member,Estimated_Salary,Contract_Type,Payment_Method,Default_or_Churn
42,Female,75000,680,36,45000.50,2,1,1,82000.00,Month-to-Month,Electronic Check,0
55,Male,110000,720,60,89000.75,1,1,1,105000.50,Two-Year,Bank Transfer,0
28,Male,48000,590,12,12500.00,1,0,0,52000.00,Month-to-Month,Electronic Check,1
34,Female,62000,640,24,31000.20,2,1,1,69000.00,One-Year,Credit Card,0
61,Female,95000,510,48,102000.00,3,1,0,98000.00,Month-to-Month,Electronic Check,1
45,Male,84000,760,40,67000.50,1,1,1,91000.20,Two-Year,Bank Transfer,0
23,Female,35000,600,6,8500.00,1,1,0,41000.00,Month-to-Month,Electronic Check,1
50,Male,125000,790,72,115000.00,2,1,1,130000.00,Two-Year,Bank Transfer,0
39,Male,71000,630,30,42000.00,2,0,1,77000.00,One-Year,Credit Card,0
48,Female,89000,580,18,54000.25,1,1,0,95000.00,Month-to-Month,Mailed Check,1
31,Male,56000,710,15,22000.00,2,1,1,61000.00,One-Year,Bank Transfer,0
58,Female,102000,660,54,98000.50,1,0,1,112000.00,Two-Year,Credit Card,0
29,Female,45000,570,9,14000.00,1,1,0,49000.00,Month-to-Month,Electronic Check,1
41,Male,78000,740,38,61000.00,2,1,1,85000.00,One-Year,Bank Transfer,0
36,Female,67000,695,28,39000.50,1,1,1,73000.00,Month-to-Month,Credit Card,0
52,Male,115000,540,42,88000.00,3,1,0,121000.00,Month-to-Month,Electronic Check,1
26,Male,42000,625,10,16500.00,1,0,1,46000.00,Month-to-Month,Credit Card,0
47,Female,86000,715,44,72000.00,2,1,1,93000.00,Two-Year,Bank Transfer,0
33,Female,59000,660,20,28000.00,1,1,1,64000.00,One-Year,Credit Card,0
64,Male,130000,780,68,124000.00,2,1,1,142000.00,Two-Year,Bank Transfer,0
38,Male,73000,560,22,48000.00,2,1,0,80000.00,Month-to-Month,Electronic Check,1
44,Female,81000,730,34,59000.00,2,1,1,88000.00,One-Year,Credit Card,0
27,Female,46000,610,14,19000.00,1,0,1,50000.00,Month-to-Month,Mailed Check,0
56,Male,108000,530,50,94000.00,3,1,0,116000.00,Month-to-Month,Electronic Check,1
35,Male,69000,705,26,37000.00,1,1,1,75000.00,One-Year,Bank Transfer,0
49,Female,92000,750,46,81000.00,2,1,1,101000.00,Two-Year,Bank Transfer,0
30,Male,51000,595,11,18000.00,1,1,0,56000.00,Month-to-Month,Electronic Check,1
53,Female,118000,670,58,105000.00,2,1,1,127000.00,Two-Year,Credit Card,0
40,Male,76000,650,32,51000.00,2,0,1,83000.00,One-Year,Bank Transfer,0
62,Female,99000,525,45,96000.00,3,1,0,107000.00,Month-to-Month,Electronic Check,1
32,Female,58000,725,18,26000.00,1,1,1,63000.00,One-Year,Credit Card,0
46,Male,85000,770,41,69000.00,2,1,1,92000.00,Two-Year,Bank Transfer,0
25,Female,38000,585,8,11000.00,1,0,0,43000.00,Month-to-Month,Electronic Check,1
57,Male,112000,740,62,93000.00,1,1,1,119000.00,Two-Year,Bank Transfer,0
37,Female,70000,685,27,41000.00,2,1,1,76000.00,Month-to-Month,Credit Card,0
51,Male,122000,550,49,91000.00,3,1,0,128000.00,Month-to-Month,Electronic Check,1
28,Female,47000,615,13,17000.00,1,1,1,51000.00,Month-to-Month,Mailed Check,0
43,Male,80000,735,35,63000.00,2,1,1,87000.00,One-Year,Bank Transfer,0
34,Male,64000,665,23,33000.00,1,0,1,70000.00,One-Year,Credit Card,0
60,Female,97000,515,47,99000.00,3,1,0,104000.00,Month-to-Month,Electronic Check,1
24,Male,36000,590,7,9500.00,1,1,0,40000.00,Month-to-Month,Electronic Check,1
48,Male,88000,755,43,74000.00,2,1,1,96000.00,Two-Year,Bank Transfer,0
36,Female,68000,675,25,38000.00,2,1,1,74000.00,Month-to-Month,Credit Card,0
54,Female,116000,560,52,95000.00,3,1,0,124000.00,Month-to-Month,Electronic Check,1
29,Male,49000,630,15,21000.00,1,0,1,54000.00,One-Year,Bank Transfer,0
59,Male,114000,765,65,108000.00,2,1,1,123000.00,Two-Year,Bank Transfer,0
31,Female,54000,580,12,16000.00,1,1,0,59000.00,Month-to-Month,Electronic Check,1
45,Female,83000,720,37,60000.00,2,1,1,90000.00,One-Year,Credit Card,0
33,Male,61000,690,21,30000.00,1,1,1,67000.00,Month-to-Month,Bank Transfer,0
63,Male,128000,535,55,109000.00,3,1,0,138000.00,Month-to-Month,Electronic Check,1"""
        with open(sample_finance, "w") as f:
            f.write(content.strip())
        logger.info(f"Auto-generated sample financial dataset at {sample_finance}")

    sample_health = os.path.join(data_dir, "sample_dataset_healthcare.csv")
    if not os.path.exists(sample_health) or os.path.getsize(sample_health) == 0:
        content = """Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
48,M,ASY,138,214,0,Normal,108,Y,1.5,Flat,1
54,F,NAP,160,195,0,Normal,130,N,1.0,Up,0
37,M,ATA,130,283,0,ST,98,N,0.0,Up,0
60,M,ASY,142,207,1,Normal,138,Y,1.8,Flat,1
49,F,NAP,124,190,0,Normal,166,N,0.0,Up,0
56,M,ASY,130,0,0,Normal,122,Y,1.0,Flat,1
51,F,ATA,140,230,0,Normal,150,N,0.0,Up,0
65,M,ASY,150,225,1,LVH,114,Y,2.2,Down,1
42,M,NAP,120,240,0,Normal,170,N,0.0,Up,0
58,F,ASY,155,270,0,ST,112,Y,2.0,Flat,1
39,M,ATA,120,204,0,Normal,145,N,0.0,Up,0
63,M,ASY,140,260,1,Normal,112,Y,3.0,Down,1
52,F,NAP,130,220,0,Normal,155,N,0.0,Up,0
47,M,ASY,135,250,0,Normal,132,Y,1.2,Flat,1
59,M,ASY,160,265,0,LVH,100,Y,2.5,Down,1
35,F,ATA,115,180,0,Normal,175,N,0.0,Up,0
61,M,ASY,145,210,1,Normal,125,Y,1.6,Flat,1
55,M,NAP,130,245,0,Normal,140,N,0.5,Up,0
44,F,ATA,125,215,0,Normal,160,N,0.0,Up,0
68,M,ASY,165,280,1,LVH,95,Y,3.2,Down,1
33,M,NAP,110,190,0,Normal,180,N,0.0,Up,0
57,F,ASY,140,240,0,ST,120,Y,1.8,Flat,1
46,M,ATA,120,230,0,Normal,150,N,0.0,Up,0
64,M,ASY,155,255,1,Normal,110,Y,2.0,Flat,1
50,F,NAP,128,210,0,Normal,162,N,0.0,Up,0
41,M,ATA,118,195,0,Normal,168,N,0.0,Up,0
66,M,ASY,170,290,1,LVH,90,Y,3.5,Down,1
53,M,ASY,135,235,0,Normal,130,Y,1.4,Flat,1
38,F,NAP,115,185,0,Normal,172,N,0.0,Up,0
62,M,ASY,150,245,1,ST,105,Y,2.2,Down,1"""
        with open(sample_health, "w") as f:
            f.write(content.strip())
        logger.info(f"Auto-generated sample healthcare dataset at {sample_health}")

    sample_ecom = os.path.join(data_dir, "sample_dataset_ecommerce.csv")
    if not os.path.exists(sample_ecom) or os.path.getsize(sample_ecom) == 0:
        content = """Customer_Age,Membership_Years,Annual_Spend,Num_Orders,Avg_Cart_Value,Return_Rate,Discount_Usage,Device_Type,Preferred_Category,Is_Loyal_Customer
32,3,4500.50,24,187.50,0.05,Medium,Mobile,Electronics,1
45,5,8900.00,48,185.40,0.02,High,Desktop,Home Goods,1
22,1,850.00,6,141.60,0.25,Low,Mobile,Fashion,0
28,2,2400.25,18,133.30,0.08,High,Mobile,Electronics,1
56,7,12500.00,62,201.60,0.01,Low,Desktop,Luxury,1
35,2,1800.00,10,180.00,0.15,Medium,Mobile,Books,0
41,4,6200.50,30,206.60,0.04,High,Desktop,Home Goods,1
19,1,450.00,4,112.50,0.35,Low,Mobile,Fashion,0
50,6,11000.00,55,200.00,0.02,Medium,Desktop,Electronics,1
29,3,3100.00,20,155.00,0.06,High,Mobile,Beauty,1
38,2,2100.00,12,175.00,0.18,Medium,Mobile,Books,0
62,8,15400.00,70,220.00,0.01,Low,Desktop,Luxury,1
26,1,1200.00,8,150.00,0.20,High,Mobile,Fashion,0
48,5,7800.00,40,195.00,0.03,Medium,Desktop,Electronics,1
33,3,3900.00,22,177.20,0.07,High,Mobile,Beauty,1
24,1,950.00,7,135.70,0.28,Low,Mobile,Fashion,0
52,6,10500.00,50,210.00,0.02,Low,Desktop,Home Goods,1
37,4,5400.00,28,192.80,0.05,High,Desktop,Electronics,1
43,3,4800.00,25,192.00,0.09,Medium,Mobile,Beauty,1
21,1,600.00,5,120.00,0.30,Low,Mobile,Fashion,0
59,7,13800.00,65,212.30,0.01,Low,Desktop,Luxury,1
30,2,2600.00,15,173.30,0.10,High,Mobile,Electronics,1
46,5,8200.00,42,195.20,0.03,Medium,Desktop,Home Goods,1
25,1,1100.00,9,122.20,0.22,Low,Mobile,Fashion,0
26,8,16200.00,75,216.00,0.01,Low,Desktop,Luxury,1
36,3,4200.00,23,182.60,0.06,High,Mobile,Beauty,1
54,6,11800.00,58,203.40,0.02,Medium,Desktop,Electronics,1
27,2,2300.00,16,143.75,0.12,High,Mobile,Fashion,1
40,4,5900.00,32,184.30,0.04,High,Desktop,Home Goods,1
60,7,14500.00,68,213.20,0.01,Low,Desktop,Luxury,1"""
        with open(sample_ecom, "w") as f:
            f.write(content.strip())
        logger.info(f"Auto-generated sample e-commerce dataset at {sample_ecom}")

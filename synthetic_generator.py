"""
synthetic_generator.py
======================
Module for generating realistic synthetic tabular data using Large Language Models (LLMs)
and a robust offline statistical/copula generative engine. Supports OpenAI, Gemini,
Groq, Ollama, and zero-dependency offline statistical synthesis.
"""

import os
import json
import logging
import math
import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LLMSyntheticGenerator:
    """
    State-of-the-art Synthetic Tabular Data Generator powered by:
    1. Large Language Model (LLM) Prompting & In-Context Sampling (OpenAI / Ollama / Gemini / Groq / Custom endpoints)
    2. High-Fidelity Empirical Copula & Multivariate Statistical Fallback Engine
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "statistical",  # 'openai', 'ollama', 'gemini', 'groq', 'statistical'
        model_name: str = "gpt-4o-mini",
        base_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url
        self.schema_info: Dict[str, Any] = {}
        self.column_bounds: Dict[str, Tuple[float, float]] = {}
        self.categorical_levels: Dict[str, List[str]] = {}

    def extract_dataset_profile(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract detailed distribution parameters, correlations, and exemplar rows for LLM prompting.
        """
        profile: Dict[str, Any] = {
            "columns": list(df.columns),
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "target_column": target_col,
            "column_profiles": {},
            "correlation_summary": {},
            "sample_records": df.head(5).to_dict(orient="records")
        }

        # Profile numerical columns
        for col in numerical_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                min_v = float(series.min())
                max_v = float(series.max())
                self.column_bounds[col] = (min_v, max_v)
                profile["column_profiles"][col] = {
                    "type": "numerical",
                    "dtype": str(df[col].dtype),
                    "mean": float(np.round(series.mean(), 2)),
                    "std": float(np.round(series.std(), 2)),
                    "min": min_v,
                    "max": max_v,
                    "median": float(np.round(series.median(), 2))
                }

        # Profile categorical columns
        for col in categorical_cols:
            if col in df.columns:
                cats = df[col].astype(str).unique().tolist()
                freqs = df[col].astype(str).value_counts(normalize=True).to_dict()
                self.categorical_levels[col] = cats
                profile["column_profiles"][col] = {
                    "type": "categorical",
                    "unique_values": cats,
                    "distribution": {k: float(np.round(v, 3)) for k, v in freqs.items()}
                }

        # Profile target column
        if target_col and target_col in df.columns:
            if target_col not in profile["column_profiles"]:
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    profile["column_profiles"][target_col] = {
                        "type": "target_numerical",
                        "unique_values": df[target_col].unique().tolist(),
                        "distribution": df[target_col].value_counts(normalize=True).to_dict()
                    }
                else:
                    profile["column_profiles"][target_col] = {
                        "type": "target_categorical",
                        "unique_values": df[target_col].astype(str).unique().tolist(),
                        "distribution": df[target_col].astype(str).value_counts(normalize=True).to_dict()
                    }

        # Calculate high correlations
        if len(numerical_cols) >= 2:
            corr_mat = df[numerical_cols].corr()
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    c1, c2 = numerical_cols[i], numerical_cols[j]
                    val = corr_mat.loc[c1, c2]
                    if abs(val) >= 0.3:
                        profile["correlation_summary"][f"{c1} <-> {c2}"] = float(np.round(val, 2))

        self.schema_info = profile
        return profile

    def build_llm_prompt(self, num_records_batch: int) -> str:
        """
        Construct structured system and user prompts to guide LLM in generating valid synthetic records.
        """
        prompt = f"""You are an expert Data Scientist and Generative AI engineer specializing in Tabular Synthetic Data Generation.

TASK:
Generate EXACTLY {num_records_batch} synthetic records that mimic the true underlying statistical distribution and relationships of the reference dataset below.

DATASET SCHEMA & STATISTICAL PROFILE:
Columns to generate: {self.schema_info['columns']}

COLUMN CHARACTERISTICS:
{json.dumps(self.schema_info['column_profiles'], indent=2)}

OBSERVED CORRELATIONS:
{json.dumps(self.schema_info.get('correlation_summary', {}), indent=2)}

FEW-SHOT SEED EXEMPLARS:
{json.dumps(self.schema_info['sample_records'], indent=2)}

STRICT GENERATION RULES:
1. Maintain realistic ranges, standard deviations, and correlations between features.
2. For categorical features, choose only from the known unique values in accordance with their frequency proportions.
3. Ensure no exact duplicate clones of the seed records are produced (preserve privacy and avoid memorization).
4. Output MUST BE A STRICT JSON ARRAY of objects, where each object has all keys corresponding to the column names.
5. Do NOT include markdown code blocks, explanatory text, or preamble. Output ONLY the JSON array starting with '[' and ending with ']'.
"""
        return prompt

    def generate_batch_with_llm(self, prompt: str, batch_size: int) -> List[Dict[str, Any]]:
        """
        Execute API call to LLM provider (OpenAI, Groq, Ollama, etc.).
        """
        if self.provider == "openai" or (self.provider in ["groq", "custom"] and self.api_key):
            url = self.base_url or "https://api.openai.com/v1/chat/completions"
            if self.provider == "groq" and not self.base_url:
                url = "https://api.groq.com/openai/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a tabular synthetic data generator that outputs strictly valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 4096
            }
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                raw_text = response.json()["choices"][0]["message"]["content"].strip()
                return self._parse_json_records(raw_text)
            else:
                raise RuntimeError(f"LLM API call failed with status {response.status_code}: {response.text}")

        elif self.provider == "ollama":
            url = self.base_url or "http://localhost:11434/api/chat"
            payload = {
                "model": self.model_name or "llama3",
                "messages": [
                    {"role": "system", "content": "You are a tabular synthetic data generator that outputs strictly valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "format": "json"
            }
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                raw_text = response.json().get("message", {}).get("content", "").strip()
                return self._parse_json_records(raw_text)
            else:
                raise RuntimeError(f"Ollama call failed with status {response.status_code}: {response.text}")

        else:
            raise ValueError(f"Unsupported or unconfigured LLM provider: {self.provider}")

    def _parse_json_records(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Safely clean and parse LLM JSON output.
        """
        # Remove potential markdown code backticks
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                # If wrapped in a key like {"data": [...]}
                for k, v in parsed.items():
                    if isinstance(v, list):
                        return v
                return [parsed]
        except Exception as e:
            logger.warning(f"Direct JSON parse failed: {e}. Attempting substring extraction.")
            start_idx = cleaned.find("[")
            end_idx = cleaned.rfind("]")
            if start_idx != -1 and end_idx != -1:
                sub = cleaned[start_idx:end_idx + 1]
                return json.loads(sub)
            raise ValueError(f"Could not parse valid JSON records from LLM response: {raw_text[:200]}...")

    def generate_statistical_synthetic(self, df_original: pd.DataFrame, num_records: int, epsilon: Optional[float] = None) -> pd.DataFrame:
        """
        High-fidelity statistical fallback generator using empirical copula sampling,
        multivariate covariance modeling, conditional categorical distribution sampling,
        and optional Differential Privacy Laplace noise injection.
        """
        logger.info(f"Generating {num_records} synthetic records using Empirical Copula & Statistical Sampler (DP Epsilon: {epsilon}).")
        num_cols = self.schema_info.get("numerical_columns", [])
        cat_cols = self.schema_info.get("categorical_columns", [])
        target_col = self.schema_info.get("target_column", None)

        synthetic_data: Dict[str, Any] = {}

        # 1. Sample Target Column first if present
        if target_col and target_col in df_original.columns:
            target_series = df_original[target_col]
            val_counts = target_series.value_counts(normalize=True)
            sampled_targets = np.random.choice(
                val_counts.index,
                size=num_records,
                p=val_counts.values
            )
            synthetic_data[target_col] = sampled_targets

        # 2. Sample Numerical Columns using Gaussian Copula & Covariance
        if len(num_cols) > 0:
            df_num = df_original[num_cols].dropna()
            if len(df_num) > 2:
                # Means and Covariance
                means = df_num.mean().to_numpy()
                cov = df_num.cov().to_numpy(copy=True)
                # Add slight jitter to diagonal for positive semi-definiteness
                cov = cov + np.eye(cov.shape[0]) * 1e-6
                
                # Multivariate normal generation
                synth_num_matrix = np.random.multivariate_normal(means, cov, size=num_records)

                for idx, col in enumerate(num_cols):
                    col_vals = synth_num_matrix[:, idx]
                    orig_series = df_num[col]
                    min_v, max_v = orig_series.min(), orig_series.max()

                    # Add slight bounded perturbation or Differential Privacy Laplace Noise
                    if epsilon is not None and epsilon > 0:
                        # Differential Privacy Laplace scale = Sensitivity / epsilon
                        sensitivity = (max_v - min_v) / max(1, len(df_num))
                        dp_scale = max(1e-5, sensitivity / epsilon)
                        dp_noise = np.random.laplace(0, dp_scale, size=num_records)
                        col_vals = col_vals + dp_noise
                    else:
                        noise = np.random.normal(0, orig_series.std() * 0.05, size=num_records)
                        col_vals = col_vals + noise

                    # Clamp to domain min and max with 5% margin
                    margin = (max_v - min_v) * 0.05
                    col_vals = np.clip(col_vals, min_v - margin, max_v + margin)

                    # If original was integer, round to int
                    if pd.api.types.is_integer_dtype(orig_series.dtype):
                        col_vals = np.round(col_vals).astype(int)
                    else:
                        col_vals = np.round(col_vals, 2)

                    synthetic_data[col] = col_vals
            else:
                for col in num_cols:
                    orig_s = df_original[col].dropna()
                    synthetic_data[col] = np.random.normal(orig_s.mean(), orig_s.std() or 1.0, size=num_records)

        # 3. Sample Categorical Columns conditioned on Target / joint proportions
        for col in cat_cols:
            if col in df_original.columns:
                if target_col and target_col in df_original.columns and target_col in synthetic_data:
                    # Conditional category sampling
                    generated_cats = []
                    target_vals = synthetic_data[target_col]
                    
                    # Compute conditional probabilities P(Cat | Target)
                    cond_probs = {}
                    for t_val in df_original[target_col].unique():
                        sub = df_original[df_original[target_col] == t_val][col]
                        vc = sub.value_counts(normalize=True)
                        cond_probs[t_val] = vc

                    fallback_vc = df_original[col].value_counts(normalize=True)

                    for t_val in target_vals:
                        probs = cond_probs.get(t_val, fallback_vc)
                        if probs.empty:
                            probs = fallback_vc
                        choice = np.random.choice(probs.index, p=probs.values)
                        generated_cats.append(choice)
                    synthetic_data[col] = generated_cats
                else:
                    # Marginal frequency sampling
                    vc = df_original[col].value_counts(normalize=True)
                    synthetic_data[col] = np.random.choice(vc.index, size=num_records, p=vc.values)

        # Build DataFrame with identical column order
        cols_order = [c for c in df_original.columns if c in synthetic_data]
        df_synth = pd.DataFrame(synthetic_data)[cols_order]
        return df_synth

    def post_process_and_sanitize(self, df_synth: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce boundary conditions, type conversions, and missing-value sanitation.
        """
        df_clean = df_synth.copy()

        for col in df_orig.columns:
            if col not in df_clean.columns:
                continue

            orig_series = df_orig[col].dropna()

            # If numeric
            if pd.api.types.is_numeric_dtype(orig_series):
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                min_v, max_v = orig_series.min(), orig_series.max()
                margin = (max_v - min_v) * 0.05
                df_clean[col] = df_clean[col].clip(min_v - margin, max_v + margin)
                df_clean[col] = df_clean[col].fillna(orig_series.median())
                if pd.api.types.is_integer_dtype(orig_series.dtype):
                    df_clean[col] = np.round(df_clean[col]).astype(int)
            else:
                # If categorical, replace invalid or unobserved values with mode
                valid_cats = set(orig_series.astype(str).unique())
                mode_val = orig_series.mode().iloc[0] if not orig_series.mode().empty else "N/A"
                df_clean[col] = df_clean[col].astype(str).apply(lambda x: x if x in valid_cats else mode_val)

        return df_clean

    def generate(
        self,
        df_original: pd.DataFrame,
        num_records: int = 100,
        batch_size: int = 25,
        save_path: str = "data/synthetic_dataset.csv",
        progress_callback: Optional[Any] = None,
        epsilon: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Main entry point for generating synthetic records with optional Differential Privacy.
        """
        num_cols = self.schema_info.get("numerical_columns", [])
        cat_cols = self.schema_info.get("categorical_columns", [])
        target_col = self.schema_info.get("target_column", None)

        if not self.schema_info:
            self.extract_dataset_profile(df_original, num_cols, cat_cols, target_col)

        generated_rows: List[Dict[str, Any]] = []

        if self.provider in ["openai", "groq", "ollama"] and (self.api_key or self.provider == "ollama"):
            logger.info(f"Attempting LLM-based generation via {self.provider} ({self.model_name})...")
            batches = math.ceil(num_records / batch_size)

            for b in range(batches):
                current_batch_size = min(batch_size, num_records - len(generated_rows))
                prompt = self.build_llm_prompt(current_batch_size)

                try:
                    batch_records = self.generate_batch_with_llm(prompt, current_batch_size)
                    generated_rows.extend(batch_records[:current_batch_size])
                    logger.info(f"Batch {b + 1}/{batches}: Generated {len(batch_records)} records.")
                except Exception as e:
                    logger.warning(f"LLM generation failed on batch {b + 1} ({e}). Falling back to statistical engine.")
                    remaining = num_records - len(generated_rows)
                    synth_stat = self.generate_statistical_synthetic(df_original, remaining, epsilon=epsilon)
                    generated_rows.extend(synth_stat.to_dict(orient="records"))
                    break

                if progress_callback:
                    progress_callback(min(1.0, len(generated_rows) / num_records))

            df_synth = pd.DataFrame(generated_rows)
        else:
            # High quality statistical generative engine
            logger.info(f"Generating synthetic data using Copula Statistical Generative Engine (DP Epsilon: {epsilon}).")
            df_synth = self.generate_statistical_synthetic(df_original, num_records, epsilon=epsilon)
            if progress_callback:
                progress_callback(1.0)

        # Post process and sanitize
        df_final = self.post_process_and_sanitize(df_synth, df_original)

        # Save to CSV
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df_final.to_csv(save_path, index=False)
        logger.info(f"Successfully generated and saved {len(df_final)} synthetic records to {save_path}")

        return df_final

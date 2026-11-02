"""
run_pipeline.py
===============
Autonomous CLI script to execute the complete end-to-end Data Science pipeline:
1. Ingests and cleans sample dataset.
2. Generates synthetic dataset.
3. Computes statistical distributions and exports comparison visuals.
4. Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost across 3 regimes (Original, Synthetic TSTR, Combined).
5. Computes SHAP explainability and exports summary/local plots.
6. Assesses Synthetic Data Quality Score & Privacy Risk Score (DCR).
7. Compiles comparison_report.pdf, evaluation_report.pdf, classification_report.txt, and synthetic_dataset.csv.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

from preprocessing import DatasetPreprocessor
from synthetic_generator import LLMSyntheticGenerator
from visualization import (
    plot_histograms_comparison,
    plot_boxplots_comparison,
    plot_correlation_heatmaps,
    plot_class_balance,
    plot_categorical_distributions,
    plot_confusion_matrices,
    plot_model_comparison_bars,
    plot_feature_importance,
    plot_privacy_distance
)
from model_training import MLPipeline
from evaluation import compute_statistical_similarity, compute_privacy_risk
from explainability import ExplainableAIEngine
from report_generator import (
    generate_classification_report_txt,
    generate_comparison_pdf,
    generate_evaluation_pdf
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def run_complete_pipeline(
    data_path: str = "data/sample_dataset.csv",
    target_col: str = "Default_or_Churn",
    num_synthetic: int = 150
):
    logger.info("=" * 70)
    logger.info("STARTING SYNTHETIC DATA GENERATION & ML BENCHMARK PIPELINE")
    logger.info("=" * 70)

    # 1. Preprocessing
    logger.info("[Step 1/7] Ingesting and preprocessing dataset...")
    prep = DatasetPreprocessor()
    df_raw = prep.load_data(data_path)
    prep.detect_column_types(df_raw, target_col=target_col)
    df_clean = prep.clean_data(df_raw)
    stats_summary = prep.get_summary_statistics(df_clean)
    logger.info(f"Dataset stats: {stats_summary['total_records']} rows, {len(prep.numerical_cols)} numerical, {len(prep.categorical_cols)} categorical.")

    # 2. Synthetic Data Generation
    logger.info(f"[Step 2/7] Synthesizing {num_synthetic} records using Generative Engine...")
    generator = LLMSyntheticGenerator(provider="statistical")
    generator.extract_dataset_profile(df_clean, prep.numerical_cols, prep.categorical_cols, target_col=target_col)
    df_synth = generator.generate(
        df_original=df_clean,
        num_records=num_synthetic,
        save_path="data/synthetic_dataset.csv"
    )
    logger.info(f"Saved synthetic dataset to data/synthetic_dataset.csv ({df_synth.shape})")

    # 3. Statistical Comparison & Visualizations
    logger.info("[Step 3/7] Generating distribution, boxplot, correlation, and class balance visuals...")
    img_hist = plot_histograms_comparison(df_clean, df_synth, prep.numerical_cols)
    img_box = plot_boxplots_comparison(df_clean, df_synth, prep.numerical_cols)
    img_corr = plot_correlation_heatmaps(df_clean, df_synth, prep.numerical_cols)
    img_cb = plot_class_balance(df_clean, df_synth, target_col)
    img_cat = plot_categorical_distributions(df_clean, df_synth, prep.categorical_cols)

    # 4. Machine Learning Pipeline (3 Regimes)
    logger.info("[Step 4/7] Running Machine Learning Pipeline across 3 Regimes...")
    X_orig_enc, y_orig_enc = prep.encode_and_scale(df_clean, target_col=target_col, fit=True)
    X_train_orig, X_test_real, y_train_orig, y_test_real = prep.split_dataset(
        X_orig_enc, y_orig_enc, test_size=0.25, random_state=42
    )
    X_synth_enc, y_synth_enc = prep.encode_and_scale(df_synth, target_col=target_col, fit=False)

    pipeline = MLPipeline(random_state=42)
    results_df = pipeline.run_full_pipeline(
        X_train_orig=X_train_orig,
        y_train_orig=y_train_orig,
        X_test_real=X_test_real,
        y_test_real=y_test_real,
        X_synth=X_synth_enc,
        y_synth=y_synth_enc
    )

    img_bars = plot_model_comparison_bars(results_df)
    cms_dict = {f"{r['Model']} ({r['Regime']})": r['Confusion_Matrix'] for r in pipeline.evaluation_results}
    img_cm = plot_confusion_matrices(cms_dict)
    feat_imp_df = pipeline.get_feature_importances(list(X_test_real.columns))
    img_feat = plot_feature_importance(feat_imp_df)

    # 5. Explainable AI (SHAP)
    logger.info("[Step 5/7] Computing SHAP Explainability & Global/Local Feature Drivers...")
    xai = ExplainableAIEngine()
    rf_key = [k for k in pipeline.trained_models if "Random Forest" in k][0]
    rf_model = pipeline.trained_models[rf_key]
    xai.compute_shap(rf_model, X_train_orig, X_test_real, model_name=rf_key)
    img_shap_sum = xai.plot_shap_summary(X_test_real)
    img_shap_loc = xai.plot_local_explanation(0, X_test_real)

    # 6. Quality & Privacy Assessment
    logger.info("[Step 6/7] Computing Synthetic Data Quality Score & Privacy Risk (DCR)...")
    quality_res = compute_statistical_similarity(df_clean, df_synth, prep.numerical_cols, prep.categorical_cols)
    privacy_res = compute_privacy_risk(X_orig_enc, X_synth_enc)
    img_dcr = plot_privacy_distance(privacy_res["DCR_Distances"])

    logger.info(f"-> Synthetic Data Quality Score: {quality_res['Quality_Score_Pct']}%")
    logger.info(f"-> Privacy Protection Index: {privacy_res['Privacy_Protection_Score_Pct']}%")
    logger.info(f"-> Exact Match Clones: {privacy_res['Exact_Clones_Count']}")

    # 7. Deliverables & PDF Reports Compilation
    logger.info("[Step 7/7] Compiling PDF reports and text deliverables...")
    generate_classification_report_txt(pipeline.evaluation_results, save_path="reports/classification_report.txt")
    
    comp_imgs = [p for p in [img_hist, img_corr, img_dcr] if p and os.path.exists(p)]
    generate_comparison_pdf(
        stats_data=stats_summary,
        quality_data=quality_res,
        privacy_data=privacy_res,
        image_paths=comp_imgs,
        save_path="reports/comparison_report.pdf"
    )

    eval_imgs = [p for p in [img_bars, img_cm, img_shap_sum] if p and os.path.exists(p)]
    generate_evaluation_pdf(
        results_df=results_df,
        image_paths=eval_imgs,
        save_path="reports/evaluation_report.pdf"
    )

    logger.info("=" * 70)
    logger.info("🎉 PIPELINE RUN COMPLETED SUCCESSFULLY!")
    logger.info("Generated Artifacts:")
    logger.info("1. data/synthetic_dataset.csv")
    logger.info("2. reports/comparison_report.pdf")
    logger.info("3. reports/evaluation_report.pdf")
    logger.info("4. reports/classification_report.txt")
    logger.info("5. visuals/ (all PNG charts)")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_complete_pipeline()

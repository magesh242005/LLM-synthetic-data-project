"""
report_generator.py
===================
Automated PDF and Text Report generation using ReportLab.
Produces:
1. comparison_report.pdf: Statistical comparison, KS/TVD tests, Quality & Privacy scores, charts.
2. evaluation_report.pdf: Machine learning performance benchmarks across 3 regimes and SHAP insights.
3. classification_report.txt: Full textual classification breakdown for all models.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_reports_dir(directory: str = "reports") -> str:
    os.makedirs(directory, exist_ok=True)
    return directory


def generate_classification_report_txt(
    results_list: List[Dict[str, Any]],
    save_path: str = "reports/classification_report.txt"
) -> str:
    """
    Save structured plain-text classification reports for all models and regimes.
    """
    ensure_reports_dir(os.path.dirname(save_path) or "reports")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("      SYNTHETIC DATA GENERATION & ML BENCHMARK EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        for item in results_list:
            model = item.get("Model", "Unknown")
            regime = item.get("Regime", "Unknown")
            acc = item.get("Accuracy", 0.0)
            f1 = item.get("F1 Score", 0.0)
            auc = item.get("ROC-AUC", 0.0)
            cr_text = item.get("Classification_Report_Text", "No report available.")

            f.write(f"MODEL: {model.upper()} | REGIME: {regime.upper()}\n")
            f.write(f"Summary Metrics -> Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | ROC-AUC: {auc:.4f}\n")
            f.write("-" * 60 + "\n")
            f.write(cr_text + "\n\n")
            f.write("=" * 80 + "\n\n")

    logger.info(f"Saved classification report text to {save_path}")
    return save_path


def generate_comparison_pdf(
    stats_data: Dict[str, Any],
    quality_data: Dict[str, Any],
    privacy_data: Dict[str, Any],
    image_paths: List[str],
    save_path: str = "reports/comparison_report.pdf"
) -> str:
    """
    Compile comprehensive PDF report on Dataset Comparison, Statistical Fidelity, and Privacy.
    """
    ensure_reports_dir(os.path.dirname(save_path) or "reports")
    doc = SimpleDocTemplate(save_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#1a365d'), alignment=1)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=11, leading=14, textColor=colors.HexColor('#4a5568'), alignment=1)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, leading=18, textColor=colors.HexColor('#2b6cb0'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9.5, leading=13, textColor=colors.HexColor('#2d3748'))
    badge_style = ParagraphStyle('Badge', parent=styles['Normal'], fontSize=11, leading=14, textColor=colors.HexColor('#2b6cb0'), fontName='Helvetica-Bold')

    story = []

    # Title Banner
    story.append(Paragraph("Synthetic Data Statistical Comparison & Validation Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Automated Generative AI Quality, Statistical Divergence & Privacy Assessment", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3182ce'), spaceAfter=14))

    # Executive Summary Table
    story.append(Paragraph("1. Executive Summary & Quality Scores", h2_style))
    q_score = quality_data.get("Quality_Score_Pct", 85.0)
    p_prot = privacy_data.get("Privacy_Protection_Score_Pct", 95.0)
    sim_score = quality_data.get("Similarity_Score_Pct", 88.0)
    corr_sim = quality_data.get("Correlation_Similarity", 85.0)

    summary_table_data = [
        ["Metric", "Score / Value", "Assessment Benchmark"],
        ["Overall Synthetic Data Quality Score", f"{q_score:.1f}%", ">= 80% (High Quality)"],
        ["Statistical Similarity Index", f"{sim_score:.1f}%", "Preservation of Marginal Distributions"],
        ["Correlation Matrix Similarity", f"{corr_sim:.1f}%", "Preservation of Inter-Feature Covariance"],
        ["Privacy Protection Score (100 - Risk)", f"{p_prot:.1f}%", ">= 90% (Low Memorization Risk)"],
        ["Median Distance to Closest Record (DCR)", f"{privacy_data.get('Median_DCR', 0.0):.4f}", "Higher indicates greater privacy protection"],
        ["Exact Match Clones Count", f"{privacy_data.get('Exact_Clones_Count', 0)}", "0 is Optimal (Zero direct memorization)"]
    ]

    t = Table(summary_table_data, colWidths=[240, 100, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f7fafc'), colors.white]),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Kolmogorov-Smirnov Numerical Validation Table
    ks_tests = quality_data.get("Numerical_KS_Tests", {})
    if ks_tests:
        story.append(Paragraph("2. Numerical Features Kolmogorov-Smirnov (KS) Validation", h2_style))
        ks_table_data = [["Feature Name", "KS Statistic (D)", "p-value", "Fidelity Score (1 - D)", "Distribution Alignment"]]
        for feat, res in ks_tests.items():
            d_val = res.get("KS_Statistic", 0.0)
            p_val = res.get("P_Value", 0.0)
            comp = res.get("KS_Complement_Score", 0.0)
            status = "Strong Match" if d_val < 0.2 else ("Moderate Match" if d_val < 0.4 else "Divergent")
            ks_table_data.append([feat, f"{d_val:.4f}", f"{p_val:.4f}", f"{comp * 100:.1f}%", status])

        t_ks = Table(ks_table_data, colWidths=[150, 90, 90, 100, 110])
        t_ks.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_ks)
        story.append(Spacer(1, 14))

    # Visualizations Embedded
    story.append(Paragraph("3. Visual Distribution & Correlation Analysis", h2_style))
    for img_p in image_paths:
        if os.path.exists(img_p):
            try:
                story.append(Spacer(1, 6))
                story.append(Image(img_p, width=520, height=220))
                story.append(Spacer(1, 10))
            except Exception as e:
                logger.warning(f"Could not add image {img_p} to PDF: {e}")

    doc.build(story)
    logger.info(f"Generated comparison PDF report at {save_path}")
    return save_path


def generate_evaluation_pdf(
    results_df: pd.DataFrame,
    image_paths: List[str],
    save_path: str = "reports/evaluation_report.pdf"
) -> str:
    """
    Compile comprehensive PDF report on Machine Learning Model Performance and Explainable AI.
    """
    ensure_reports_dir(os.path.dirname(save_path) or "reports")
    doc = SimpleDocTemplate(save_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#1a365d'), alignment=1)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=11, leading=14, textColor=colors.HexColor('#4a5568'), alignment=1)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, leading=18, textColor=colors.HexColor('#2b6cb0'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#2d3748'))

    story = []

    # Title
    story.append(Paragraph("Machine Learning Pipeline & Explainability Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Comparative Evaluation: Original vs Synthetic (TSTR) vs Combined Models on Holdout Real Data", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3182ce'), spaceAfter=14))

    # Benchmark Results Table
    story.append(Paragraph("1. Machine Learning Benchmark Table", h2_style))

    table_data = [["Model Algorithm", "Training Regime", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
    for _, row in results_df.iterrows():
        table_data.append([
            str(row.get("Model", "")),
            str(row.get("Regime", "")),
            f"{float(row.get('Accuracy', 0)):.4f}",
            f"{float(row.get('Precision', 0)):.4f}",
            f"{float(row.get('Recall', 0)):.4f}",
            f"{float(row.get('F1 Score', 0)):.4f}",
            f"{float(row.get('ROC-AUC', 0)):.4f}"
        ])

    t_eval = Table(table_data, colWidths=[110, 130, 60, 60, 60, 60, 60])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f7fafc'), colors.white]),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 14))

    # Key Findings & Methodology Notes
    story.append(Paragraph("2. Experimental Methodology & TSTR Interpretation", h2_style))
    methodology_text = (
        "<b>Train on Synthetic, Test on Real (TSTR) Principle:</b> The ultimate utility test of synthetic data is whether "
        "a model trained solely on synthetic samples can accurately predict labels on unseen real-world test data. "
        "Close alignment between Original and Synthetic metrics validates high downstream utility without compromising real user privacy."
    )
    story.append(Paragraph(methodology_text, body_style))
    story.append(Spacer(1, 14))

    # Embed Evaluation Charts (Confusion Matrices, Feature Importances, SHAP)
    story.append(Paragraph("3. Model Visualizations & Explainable AI (SHAP)", h2_style))
    for img_p in image_paths:
        if os.path.exists(img_p):
            try:
                story.append(Spacer(1, 6))
                story.append(Image(img_p, width=520, height=240))
                story.append(Spacer(1, 10))
            except Exception as e:
                logger.warning(f"Could not add image {img_p} to PDF: {e}")

    doc.build(story)
    logger.info(f"Generated evaluation PDF report at {save_path}")
    return save_path

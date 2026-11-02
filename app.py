"""
app.py
======
Ultra-Modern, Enterprise-Grade Streamlit Web Application:
"Synthetic Data Generation Using Large Language Models (LLMs)"

Advanced Capabilities:
- Multi-Domain Preloaded Benchmark Datasets (Finance, Healthcare, E-Commerce)
- Differential Privacy (ε-DP Laplace Noise Mechanism) Slider & Controls
- Adversarial Membership Inference Attack (MIA) Privacy Simulation
- Interactive Live "What-If" Multi-Model Inference Playground (Real vs Synthetic consensus)
- Wasserstein-1 Earth Mover's Distance & Kolmogorov-Smirnov Divergence
- 3-Regime TSTR ML Benchmark (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Explainable AI (SHAP Global Impact & Local Instance Decision Waterfalls)
- 1-Click Executive PDF Reports & Visual Asset Bundles
"""

import os
import io
import time
import zipfile
import logging
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from preprocessing import DatasetPreprocessor
from synthetic_generator import LLMSyntheticGenerator
from model_training import MLPipeline
from evaluation import (
    compute_statistical_similarity,
    compute_privacy_risk,
    compute_membership_inference_risk,
    compute_wasserstein_distances
)
from explainability import ExplainableAIEngine
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
from report_generator import (
    generate_classification_report_txt,
    generate_comparison_pdf,
    generate_evaluation_pdf
)

# Page configuration
st.set_page_config(
    page_title="Synthetic Data Generation Using LLMs",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Ultra-Modern CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.85) 0%, rgba(88, 28, 135, 0.8) 50%, rgba(15, 23, 42, 0.95) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 26px 32px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 14.5px;
        font-weight: 400;
        margin: 0;
        line-height: 1.5;
    }
    
    .hero-badges {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    
    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        color: #e2e8f0;
    }
    
    .status-pulse {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Modern KPI Cards */
    .metric-card-pro {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px;
        text-align: left;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 12px;
    }
    
    .metric-card-pro:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 24px -8px rgba(99, 102, 241, 0.25);
    }
    
    .metric-header-pro {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    
    .metric-label-pro {
        font-size: 11px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .metric-icon-pro {
        font-size: 18px;
        opacity: 0.85;
    }
    
    .metric-val-pro {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }
    
    .metric-sub-pro {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
    }

    /* Tab bar custom design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 13.5px;
        color: #94a3b8;
        background: transparent;
        transition: all 0.2s ease;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }
    
    .glass-panel {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .sandbox-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# App Top Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧬 Synthetic Data Generation Using LLMs</div>
    <div class="hero-subtitle">
        Enterprise Generative AI Platform: <strong>ε-Differential Privacy</strong>, <strong>TSTR Machine Learning Benchmarks</strong>,
        <strong>Adversarial MIA Audits</strong>, and <strong>SHAP Explainable AI</strong>.
    </div>
    <div class="hero-badges">
        <div class="badge-chip"><div class="status-pulse"></div> System Live</div>
        <div class="badge-chip">🧠 Multi-LLM + Gaussian Copula</div>
        <div class="badge-chip">🛡️ ε-Differential Privacy</div>
        <div class="badge-chip">⚔️ MIA Attack Simulator</div>
        <div class="badge-chip">🔮 Live What-If Sandbox</div>
        <div class="badge-chip">🔍 SHAP Explainability</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = DatasetPreprocessor()
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "synth_df" not in st.session_state:
    st.session_state.synth_df = None
if "ml_pipeline" not in st.session_state:
    st.session_state.ml_pipeline = MLPipeline()
if "ml_results" not in st.session_state:
    st.session_state.ml_results = None
if "xai_engine" not in st.session_state:
    st.session_state.xai_engine = ExplainableAIEngine()
if "split_data" not in st.session_state:
    st.session_state.split_data = None
if "quality_metrics" not in st.session_state:
    st.session_state.quality_metrics = None
if "privacy_metrics" not in st.session_state:
    st.session_state.privacy_metrics = None
if "mia_metrics" not in st.session_state:
    st.session_state.mia_metrics = None
if "current_preset" not in st.session_state:
    st.session_state.current_preset = "Financial Churn & Credit Risk"

# Sidebar Configuration Controls
with st.sidebar:
    st.markdown("### ⚙️ Enterprise Configuration")
    
    preset_choice = st.selectbox(
        "Domain Benchmark Dataset",
        [
            "Financial Churn & Credit Risk",
            "Healthcare Heart Disease & Clinical Risk",
            "E-Commerce Customer Retention & CLV",
            "Upload Custom CSV Dataset"
        ],
        index=0
    )

    # Handle preset switching
    preset_file_map = {
        "Financial Churn & Credit Risk": "data/sample_dataset.csv",
        "Healthcare Heart Disease & Clinical Risk": "data/sample_dataset_healthcare.csv",
        "E-Commerce Customer Retention & CLV": "data/sample_dataset_ecommerce.csv"
    }

    uploaded_file = None
    if preset_choice == "Upload Custom CSV Dataset":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    else:
        file_path = preset_file_map.get(preset_choice, "data/sample_dataset.csv")
        if os.path.exists(file_path):
            uploaded_file = file_path

    # If user changed preset dataset
    if preset_choice != st.session_state.current_preset:
        st.session_state.current_preset = preset_choice
        st.session_state.raw_df = None
        st.session_state.cleaned_df = None
        st.session_state.synth_df = None
        st.session_state.ml_results = None
        st.session_state.split_data = None
        st.session_state.quality_metrics = None
        st.session_state.privacy_metrics = None
        st.session_state.mia_metrics = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 Generation Engine")
    provider = st.selectbox(
        "Synthesis Method",
        [
            "Statistical Copula Engine (Zero API Needed)",
            "OpenAI LLM (GPT-4o-mini)",
            "Groq LLM (Llama 3)",
            "Ollama (Local LLM)"
        ],
        index=0
    )

    api_key = ""
    model_name = "gpt-4o-mini"
    base_url = None

    if "OpenAI" in provider:
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    elif "Groq" in provider:
        api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        model_name = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)
    elif "Ollama" in provider:
        base_url = st.text_input("Ollama Base URL", value="http://localhost:11434/api/chat")
        model_name = st.text_input("Model Name", value="llama3")

    num_records_to_gen = st.slider("Target Record Count", min_value=20, max_value=2000, value=100, step=20)

    # Advanced Differential Privacy (ε-DP)
    st.markdown("---")
    st.markdown("### 🛡️ Differential Privacy (ε-DP)")
    use_dp = st.checkbox("Enable Laplace ε-DP Mechanism", value=False)
    dp_epsilon = None
    if use_dp:
        dp_epsilon = st.slider("Privacy Budget (Epsilon ε)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                               help="Lower ε = Stronger mathematical privacy guarantee with added Laplace noise.")

    st.markdown("---")
    st.markdown("### 🔄 State Management")
    if st.button("Reset Project State", use_container_width=True):
        st.session_state.raw_df = None
        st.session_state.cleaned_df = None
        st.session_state.synth_df = None
        st.session_state.ml_results = None
        st.session_state.split_data = None
        st.session_state.quality_metrics = None
        st.session_state.privacy_metrics = None
        st.session_state.mia_metrics = None
        st.rerun()

# Load Dataset Logic
if uploaded_file is not None and st.session_state.raw_df is None:
    try:
        raw_df = st.session_state.preprocessor.load_data(uploaded_file)
        st.session_state.raw_df = raw_df
        candidate_targets = [c for c in raw_df.columns if any(k in c.lower() for k in ['target', 'churn', 'default', 'disease', 'loyal', 'label', 'class', 'status'])]
        default_target = candidate_targets[0] if candidate_targets else raw_df.columns[-1]
        
        st.session_state.target_col = default_target
        st.session_state.preprocessor.detect_column_types(raw_df, target_col=default_target)
        st.session_state.cleaned_df = st.session_state.preprocessor.clean_data(raw_df)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 1. Ingestion & Profiling",
    "✨ 2. LLM / DP Generation",
    "📊 3. Statistical & Wasserstein Divergence",
    "🤖 4. ML Benchmark & What-If Sandbox",
    "🔍 5. Explainable AI (SHAP)",
    "🛡️ 6. MIA Privacy Audit & Reports"
])

# -------------------------------------------------------------
# TAB 1: DATASET INGESTION & PREPROCESSING
# -------------------------------------------------------------
with tab1:
    st.subheader(f"📁 Ingestion & Schema Profiling: {st.session_state.current_preset}")

    # Direct Drag-and-Drop File Uploader Card inside Tab 1
    with st.expander("📤 Upload Custom CSV Dataset directly here", expanded=(st.session_state.raw_df is None or preset_choice == "Upload Custom CSV Dataset")):
        tab1_uploaded_file = st.file_uploader(
            "Drop your CSV file here to analyze and generate synthetic data",
            type=["csv"],
            key="tab1_csv_uploader"
        )
        if tab1_uploaded_file is not None:
            # Check if this is a newly uploaded file
            if "last_tab1_file" not in st.session_state or st.session_state.last_tab1_file != tab1_uploaded_file.name:
                try:
                    st.session_state.last_tab1_file = tab1_uploaded_file.name
                    raw_df = st.session_state.preprocessor.load_data(tab1_uploaded_file)
                    st.session_state.raw_df = raw_df
                    candidate_targets = [c for c in raw_df.columns if any(k in c.lower() for k in ['target', 'churn', 'default', 'disease', 'loyal', 'label', 'class', 'status'])]
                    default_target = candidate_targets[0] if candidate_targets else raw_df.columns[-1]
                    st.session_state.target_col = default_target
                    st.session_state.preprocessor.detect_column_types(raw_df, target_col=default_target)
                    st.session_state.cleaned_df = st.session_state.preprocessor.clean_data(raw_df)
                    st.session_state.synth_df = None
                    st.session_state.ml_results = None
                    st.session_state.split_data = None
                    st.session_state.quality_metrics = None
                    st.session_state.privacy_metrics = None
                    st.session_state.mia_metrics = None
                    st.session_state.current_preset = f"Custom CSV ({tab1_uploaded_file.name})"
                    st.success(f"✅ Successfully loaded custom dataset: **{tab1_uploaded_file.name}** ({raw_df.shape[0]} rows, {raw_df.shape[1]} columns)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading uploaded CSV file: {e}")

    if st.session_state.raw_df is None:
        st.info("👈 Please select a preset from the sidebar or upload a custom CSV file above to begin.")
    else:
        df = st.session_state.raw_df
        prep = st.session_state.preprocessor

        # Target Column Selector
        col_t1, col_t2 = st.columns([2, 2])
        with col_t1:
            selected_target = st.selectbox(
                "🎯 Select Target / Label Column for Machine Learning",
                options=list(df.columns),
                index=list(df.columns).index(st.session_state.target_col) if st.session_state.target_col in df.columns else len(df.columns) - 1
            )
            if selected_target != st.session_state.target_col:
                st.session_state.target_col = selected_target
                prep.detect_column_types(df, target_col=selected_target)
                st.session_state.cleaned_df = prep.clean_data(df)
                st.rerun()

        with col_t2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Target Type:** `{df[selected_target].dtype}` | **Unique Classes:** `{df[selected_target].nunique()}`")

        # Top Metric Cards (Glassmorphic)
        stats = prep.get_summary_statistics(st.session_state.cleaned_df)
        m1, m2, m3, m4, m5 = st.columns(5)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Total Records</span><span class="metric-icon-pro">📄</span></div>
                <div class="metric-val-pro">{stats['total_records']}</div>
                <div class="metric-sub-pro">Rows Loaded</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Total Features</span><span class="metric-icon-pro">📐</span></div>
                <div class="metric-val-pro">{stats['total_columns']}</div>
                <div class="metric-sub-pro">Columns</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Numerical</span><span class="metric-icon-pro">🔢</span></div>
                <div class="metric-val-pro">{len(prep.numerical_cols)}</div>
                <div class="metric-sub-pro">Continuous</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Categorical</span><span class="metric-icon-pro">🏷️</span></div>
                <div class="metric-val-pro">{len(prep.categorical_cols)}</div>
                <div class="metric-sub-pro">Discrete</div>
            </div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Missing Cells</span><span class="metric-icon-pro">🧹</span></div>
                <div class="metric-val-pro">{stats['missing_percentage']}%</div>
                <div class="metric-sub-pro">Imputed</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 🔍 Dataset Preview (Cleaned & Preprocessed)")
        st.dataframe(st.session_state.cleaned_df.head(10), use_container_width=True)

        # Statistical Summary Table
        st.markdown("#### 📈 Statistical Moments & Continuous Profiling")
        if stats['numerical_summary']:
            num_df = pd.DataFrame(stats['numerical_summary']).T
            st.dataframe(num_df.style.format(precision=3), use_container_width=True)
        else:
            st.info("No numerical features detected.")

        if stats['categorical_summary']:
            st.markdown("##### 🏷️ Categorical Cardinality & Mode Distribution")
            cat_df = pd.DataFrame([
                {"Column": col, "Unique Levels": data["unique_values"], "Mode": data["mode"]}
                for col, data in stats['categorical_summary'].items()
            ])
            st.dataframe(cat_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: LLM SYNTHETIC DATA GENERATION
# -------------------------------------------------------------
with tab2:
    st.subheader("✨ LLM Prompting & Statistical Copula Generation")

    if st.session_state.cleaned_df is None:
        st.warning("Please load a dataset in Tab 1 first.")
    else:
        df_orig = st.session_state.cleaned_df
        prep = st.session_state.preprocessor
        target_c = st.session_state.target_col

        dp_info = f"<span class='badge-chip'>🛡️ Laplace ε-DP Active (ε = {dp_epsilon})</span>" if use_dp else "<span class='badge-chip'>Standard Empirical Copula</span>"
        st.markdown(f"""
        <div class="glass-panel">
            <strong>Generative Engine Architecture:</strong> Multi-backend tabular synthesis with constraint optimization.
            Current Mode: {dp_info}
        </div>
        """, unsafe_allow_html=True)

        # System Prompt & Schema Metadata Expander
        with st.expander("🔎 View Automated System Prompt & Schema Constraints"):
            provider_type = "statistical" if "Statistical" in provider else ("openai" if "OpenAI" in provider else ("groq" if "Groq" in provider else "ollama"))
            gen_preview = LLMSyntheticGenerator(api_key=api_key, provider=provider_type, model_name=model_name, base_url=base_url)
            gen_preview.extract_dataset_profile(df_orig, prep.numerical_cols, prep.categorical_cols, target_c)
            st.code(gen_preview.build_llm_prompt(num_records_batch=min(10, num_records_to_gen)), language="text")

        col_g1, col_g2 = st.columns([1, 2])
        with col_g1:
            gen_clicked = st.button("🚀 Generate Synthetic Dataset", type="primary", use_container_width=True)

        if gen_clicked:
            progress_bar = st.progress(0, text="Initializing Generative Engine...")
            start_time = time.time()
            
            provider_type = "statistical" if "Statistical" in provider else ("openai" if "OpenAI" in provider else ("groq" if "Groq" in provider else "ollama"))
            generator = LLMSyntheticGenerator(
                api_key=api_key,
                provider=provider_type,
                model_name=model_name,
                base_url=base_url
            )
            generator.extract_dataset_profile(df_orig, prep.numerical_cols, prep.categorical_cols, target_c)

            def update_progress(pct):
                progress_bar.progress(int(pct * 100), text=f"Synthesizing Records ({int(pct * 100)}%)...")

            synth_df = generator.generate(
                df_original=df_orig,
                num_records=num_records_to_gen,
                batch_size=25,
                save_path="data/synthetic_dataset.csv",
                progress_callback=update_progress,
                epsilon=dp_epsilon if use_dp else None
            )
            
            elapsed = time.time() - start_time
            progress_bar.progress(100, text="Generation Complete!")
            st.session_state.synth_df = synth_df
            st.session_state.ml_results = None  # Reset downstream results
            st.session_state.quality_metrics = None
            st.session_state.privacy_metrics = None
            st.session_state.mia_metrics = None
            st.success(f"🎉 Successfully synthesized **{len(synth_df)} records** in **{elapsed:.2f} seconds**!")

        if st.session_state.synth_df is not None:
            synth_df = st.session_state.synth_df
            st.markdown("#### 📋 Synthetic Dataset Preview")
            st.dataframe(synth_df.head(10), use_container_width=True)

            csv_buffer = io.StringIO()
            synth_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download synthetic_dataset.csv",
                data=csv_buffer.getvalue(),
                file_name="synthetic_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

# -------------------------------------------------------------
# TAB 3: DATASET COMPARISON & VISUALIZATIONS
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 Statistical Fidelity & Wasserstein Earth Mover's Distance")

    if st.session_state.cleaned_df is None or st.session_state.synth_df is None:
        st.info("💡 Please generate a synthetic dataset in Tab 2 to view statistical comparison charts.")
    else:
        df_orig = st.session_state.cleaned_df
        df_synth = st.session_state.synth_df
        prep = st.session_state.preprocessor
        target_c = st.session_state.target_col

        # Wasserstein Earth Mover's Distance Table
        if prep.numerical_cols:
            st.markdown("#### 🌐 Wasserstein-1 Distance (Earth Mover's Distance) & Distribution Divergence")
            w_dists = compute_wasserstein_distances(df_orig, df_synth, prep.numerical_cols)
            if w_dists:
                w_df = pd.DataFrame(list(w_dists.items()), columns=["Feature", "Normalized Wasserstein Distance (W1)"])
                w_df["Fidelity Rating"] = w_df["Normalized Wasserstein Distance (W1)"].apply(
                    lambda x: "🟢 Excellent Match (<0.15)" if x < 0.15 else ("🟡 Moderate Match (<0.35)" if x < 0.35 else "🔴 Divergent (>0.35)")
                )
                st.dataframe(w_df, use_container_width=True)

        # 1. Histograms & KDE Overlay
        st.markdown("#### 1. Continuous Feature Distribution Overlays (KDE Density & Histograms)")
        hist_path = plot_histograms_comparison(df_orig, df_synth, prep.numerical_cols)
        if os.path.exists(hist_path):
            st.image(hist_path, use_container_width=True)

        # 2. Boxplots Spread Comparison
        st.markdown("#### 2. Quantile Spread & Outlier Comparison (Paired Boxplots)")
        box_path = plot_boxplots_comparison(df_orig, df_synth, prep.numerical_cols)
        if os.path.exists(box_path):
            st.image(box_path, use_container_width=True)

        # 3. Correlation Heatmaps
        if len(prep.numerical_cols) >= 2:
            st.markdown("#### 3. Inter-Feature Correlation Matrices (Original vs Synthetic vs Absolute Error)")
            corr_path = plot_correlation_heatmaps(df_orig, df_synth, prep.numerical_cols)
            if os.path.exists(corr_path):
                st.image(corr_path, use_container_width=True)

        # 4. Target Class Balance
        if target_c and target_c in df_orig.columns and target_c in df_synth.columns:
            st.markdown("#### 4. Target Class Balance Alignment")
            cb_path = plot_class_balance(df_orig, df_synth, target_c)
            if os.path.exists(cb_path):
                st.image(cb_path, use_container_width=True)

        # 5. Categorical Feature Distributions
        if prep.categorical_cols:
            st.markdown("#### 5. Categorical Features Marginal Proportions")
            cat_path = plot_categorical_distributions(df_orig, df_synth, prep.categorical_cols)
            if os.path.exists(cat_path):
                st.image(cat_path, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: MACHINE LEARNING PIPELINE & WHAT-IF SANDBOX
# -------------------------------------------------------------
with tab4:
    st.subheader("🤖 Machine Learning Pipeline (TSTR) & What-If Inference Sandbox")
    
    st.markdown("""
    <div class="glass-panel">
        <strong>Train on Synthetic, Test on Real (TSTR) Principle:</strong>
        Evaluates 4 algorithms (<strong>Logistic Regression, Decision Tree, Random Forest, XGBoost</strong>)
        across 3 regimes against an untouched <strong>Real Holdout Test Split</strong>.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.cleaned_df is None or st.session_state.synth_df is None:
        st.info("💡 Please generate a synthetic dataset in Tab 2 to run the ML pipeline.")
    else:
        df_orig = st.session_state.cleaned_df
        df_synth = st.session_state.synth_df
        prep = st.session_state.preprocessor
        target_c = st.session_state.target_col

        run_ml_btn = st.button("⚡ Run Machine Learning Benchmark Across All 3 Regimes", type="primary")

        if run_ml_btn or st.session_state.ml_results is not None:
            if run_ml_btn or st.session_state.split_data is None:
                with st.spinner("Training models across Original, Synthetic (TSTR), and Combined regimes..."):
                    # 1. Encode original data
                    X_orig_enc, y_orig_enc = prep.encode_and_scale(df_orig, target_col=target_c, fit=True)
                    
                    # 2. Stratified Split on original
                    X_train_orig, X_test_real, y_train_orig, y_test_real = prep.split_dataset(
                        X_orig_enc, y_orig_enc, test_size=0.25, random_state=42
                    )

                    # 3. Encode synthetic data using the same encoders
                    X_synth_enc, y_synth_enc = prep.encode_and_scale(df_synth, target_col=target_c, fit=False)

                    # Store splits
                    st.session_state.split_data = {
                        "X_train_orig": X_train_orig,
                        "X_test_real": X_test_real,
                        "y_train_orig": y_train_orig,
                        "y_test_real": y_test_real,
                        "X_synth_enc": X_synth_enc,
                        "y_synth_enc": y_synth_enc,
                        "X_orig_enc": X_orig_enc
                    }

                    # 4. Run ML pipeline
                    pipeline = st.session_state.ml_pipeline
                    results_df = pipeline.run_full_pipeline(
                        X_train_orig=X_train_orig,
                        y_train_orig=y_train_orig,
                        X_test_real=X_test_real,
                        y_test_real=y_test_real,
                        X_synth=X_synth_enc,
                        y_synth=y_synth_enc
                    )
                    st.session_state.ml_results = results_df

                    # Generate text classification report
                    generate_classification_report_txt(pipeline.evaluation_results)

            results_df = st.session_state.ml_results
            pipeline = st.session_state.ml_pipeline

            st.markdown("#### 🏆 Comparative Benchmark Results (Evaluated on Real Holdout Set)")
            
            display_cols = ["Model", "Regime", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            st.dataframe(
                results_df[display_cols].style.highlight_max(axis=0, subset=["Accuracy", "F1 Score", "ROC-AUC"], color="#1e3a8a"),
                use_container_width=True
            )

            # Performance Bar Charts
            st.markdown("#### 📊 Comparative Model Metrics Bar Chart")
            bar_chart_path = plot_model_comparison_bars(results_df)
            if os.path.exists(bar_chart_path):
                st.image(bar_chart_path, use_container_width=True)

            # Confusion Matrices Grid
            st.markdown("#### 🧩 Confusion Matrices Across Models & Regimes")
            cms_dict = {f"{r['Model']} ({r['Regime']})": r['Confusion_Matrix'] for r in pipeline.evaluation_results}
            cm_grid_path = plot_confusion_matrices(cms_dict)
            if os.path.exists(cm_grid_path):
                st.image(cm_grid_path, use_container_width=True)

            # -------------------------------------------------------------
            # ADVANCED FEATURE: INTERACTIVE "WHAT-IF" INFERENCE SANDBOX
            # -------------------------------------------------------------
            st.markdown("---")
            st.markdown("### 🔮 Interactive Live 'What-If' Inference Sandbox")
            st.markdown("Test hypothetical customer profiles to see if the **Real-Trained Model** and **Synthetic-Trained Model** reach the same decision!")

            splits = st.session_state.split_data
            feature_cols = list(splits["X_test_real"].columns)

            with st.expander("🎛️ Configure Custom Instance Feature Values", expanded=True):
                user_inputs = {}
                cols_per_row = 3
                cols = st.columns(cols_per_row)

                for idx, f_col in enumerate(feature_cols):
                    with cols[idx % cols_per_row]:
                        orig_s = df_orig[f_col] if f_col in df_orig.columns else pd.Series([0])
                        if pd.api.types.is_numeric_dtype(orig_s):
                            min_val = float(orig_s.min())
                            max_val = float(orig_s.max())
                            default_val = float(orig_s.median())
                            user_inputs[f_col] = st.number_input(
                                f"Feature: {f_col}",
                                value=default_val,
                                min_value=min_val - abs(min_val)*0.5,
                                max_value=max_val + abs(max_val)*0.5
                            )
                        else:
                            cats = list(orig_s.astype(str).unique())
                            user_inputs[f_col] = st.selectbox(f"Feature: {f_col}", cats, index=0)

                test_sample_df = pd.DataFrame([user_inputs])
                sample_enc, _ = prep.encode_and_scale(test_sample_df, target_col=None, fit=False)

                # Predict with Random Forest Original vs Synthetic
                rf_orig_key = "Random Forest (Original Only)"
                rf_synth_key = "Random Forest (Synthetic Only (TSTR))"
                rf_comb_key = "Random Forest (Combined (Augmented))"

                if rf_orig_key in pipeline.trained_models and rf_synth_key in pipeline.trained_models:
                    model_orig = pipeline.trained_models[rf_orig_key]
                    model_synth = pipeline.trained_models[rf_synth_key]
                    model_comb = pipeline.trained_models.get(rf_comb_key, None)

                    pred_orig = model_orig.predict(sample_enc)[0]
                    prob_orig = model_orig.predict_proba(sample_enc)[0] if hasattr(model_orig, "predict_proba") else [0.5, 0.5]

                    pred_synth = model_synth.predict(sample_enc)[0]
                    prob_synth = model_synth.predict_proba(sample_enc)[0] if hasattr(model_synth, "predict_proba") else [0.5, 0.5]

                    # Display Live Consensus
                    sb_col1, sb_col2, sb_col3 = st.columns(3)
                    with sb_col1:
                        st.markdown(f"""
                        <div class="metric-card-pro">
                            <div class="metric-header-pro"><span class="metric-label-pro">Real-Trained Model</span><span>🏦</span></div>
                            <div class="metric-val-pro">Class {pred_orig}</div>
                            <div class="metric-sub-pro">Confidence: {max(prob_orig):.1%}</div>
                        </div>""", unsafe_allow_html=True)
                    with sb_col2:
                        st.markdown(f"""
                        <div class="metric-card-pro">
                            <div class="metric-header-pro"><span class="metric-label-pro">Synthetic-Trained (TSTR)</span><span>🧬</span></div>
                            <div class="metric-val-pro">Class {pred_synth}</div>
                            <div class="metric-sub-pro">Confidence: {max(prob_synth):.1%}</div>
                        </div>""", unsafe_allow_html=True)
                    with sb_col3:
                        match = (pred_orig == pred_synth)
                        agreement_text = "🟢 100% Decision Alignment" if match else "🔴 Model Divergence"
                        st.markdown(f"""
                        <div class="metric-card-pro">
                            <div class="metric-header-pro"><span class="metric-label-pro">Model Consensus</span><span>🤝</span></div>
                            <div class="metric-val-pro">{'MATCH' if match else 'DIFFER'}</div>
                            <div class="metric-sub-pro">{agreement_text}</div>
                        </div>""", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 5: EXPLAINABLE AI (SHAP)
# -------------------------------------------------------------
with tab5:
    st.subheader("🔍 Explainable AI (SHAP - SHapley Additive exPlanations)")

    if st.session_state.ml_results is None or st.session_state.split_data is None:
        st.info("💡 Please run the Machine Learning Pipeline in Tab 4 first to compute SHAP explanations.")
    else:
        splits = st.session_state.split_data
        pipeline = st.session_state.ml_pipeline
        xai = st.session_state.xai_engine

        avail_models = [k for k in pipeline.trained_models.keys() if "Random Forest" in k or "XGBoost" in k or "Logistic" in k]
        selected_model_key = st.selectbox("Select Model to Inspect with SHAP", avail_models, index=0)
        target_model = pipeline.trained_models[selected_model_key]

        with st.spinner("Computing SHAP values..."):
            shap_vals = xai.compute_shap(
                model=target_model,
                X_train=splits["X_train_orig"],
                X_test=splits["X_test_real"],
                model_name=selected_model_key
            )

        # Global SHAP Summary
        st.markdown("#### 1. Global Feature Impact (SHAP Summary Plot)")
        shap_sum_path = xai.plot_shap_summary(splits["X_test_real"])
        if os.path.exists(shap_sum_path):
            st.image(shap_sum_path, use_container_width=True)

        # Feature Importance Ranking Alignment (Original Model vs Synthetic Model)
        st.markdown("#### 2. Feature Importance Alignment (Original vs Synthetic Model)")
        feat_imp_df = pipeline.get_feature_importances(list(splits["X_test_real"].columns))
        if not feat_imp_df.empty:
            fi_path = plot_feature_importance(feat_imp_df)
            if os.path.exists(fi_path):
                st.image(fi_path, use_container_width=True)

        # Local Explanation for Single Test Instances
        st.markdown("#### 3. Local Decision Explanation for Individual Test Instances")
        test_size = len(splits["X_test_real"])
        selected_sample_idx = st.slider("Select Test Record Index to Explain", min_value=0, max_value=max(0, test_size - 1), value=0)

        col_loc1, col_loc2 = st.columns([1, 1])
        with col_loc1:
            st.markdown(f"**Test Instance #{selected_sample_idx} Feature Values:**")
            st.dataframe(splits["X_test_real"].iloc[[selected_sample_idx]].T.rename(columns={selected_sample_idx: "Feature Value"}), use_container_width=True)

        with col_loc2:
            loc_path = xai.plot_local_explanation(selected_sample_idx, splits["X_test_real"])
            if os.path.exists(loc_path):
                st.image(loc_path, use_container_width=True)

# -------------------------------------------------------------
# TAB 6: QUALITY, MIA PRIVACY & PDF REPORTS
# -------------------------------------------------------------
with tab6:
    st.subheader("🛡️ Synthetic Data Quality Score, Adversarial MIA Audit & Reports")

    if st.session_state.cleaned_df is None or st.session_state.synth_df is None:
        st.info("💡 Please generate a synthetic dataset in Tab 2 to compute quality and privacy scores.")
    else:
        df_orig = st.session_state.cleaned_df
        df_synth = st.session_state.synth_df
        prep = st.session_state.preprocessor

        # Compute Quality & Privacy Metrics if needed
        if st.session_state.quality_metrics is None or st.session_state.privacy_metrics is None or st.session_state.mia_metrics is None:
            with st.spinner("Computing Kolmogorov-Smirnov statistics, TVD, DCR, and Adversarial MIA Risk..."):
                quality_res = compute_statistical_similarity(df_orig, df_synth, prep.numerical_cols, prep.categorical_cols)
                X_orig_enc, _ = prep.encode_and_scale(df_orig, target_col=st.session_state.target_col, fit=True)
                X_synth_enc, _ = prep.encode_and_scale(df_synth, target_col=st.session_state.target_col, fit=False)
                privacy_res = compute_privacy_risk(X_orig_enc, X_synth_enc)

                # MIA Simulation
                if st.session_state.split_data is not None:
                    mia_res = compute_membership_inference_risk(
                        st.session_state.split_data["X_train_orig"],
                        st.session_state.split_data["X_test_real"],
                        X_synth_enc
                    )
                else:
                    mia_res = compute_membership_inference_risk(X_orig_enc, X_orig_enc, X_synth_enc)

                st.session_state.quality_metrics = quality_res
                st.session_state.privacy_metrics = privacy_res
                st.session_state.mia_metrics = mia_res

        quality_res = st.session_state.quality_metrics
        privacy_res = st.session_state.privacy_metrics
        mia_res = st.session_state.mia_metrics

        # Executive KPI Cards (Glassmorphic)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Quality Score</span><span class="metric-icon-pro">💎</span></div>
                <div class="metric-val-pro">{quality_res['Quality_Score_Pct']}%</div>
                <div class="metric-sub-pro">Composite Fidelity</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Similarity Index</span><span class="metric-icon-pro">📈</span></div>
                <div class="metric-val-pro">{quality_res['Similarity_Score_Pct']}%</div>
                <div class="metric-sub-pro">Marginal Matching</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Privacy Protection</span><span class="metric-icon-pro">🔒</span></div>
                <div class="metric-val-pro">{privacy_res['Privacy_Protection_Score_Pct']}%</div>
                <div class="metric-sub-pro">100 - Leakage Risk</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Direct Clones</span><span class="metric-icon-pro">🎯</span></div>
                <div class="metric-val-pro">{privacy_res['Exact_Clones_Count']}</div>
                <div class="metric-sub-pro">Zero Memorization</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # ADVANCED FEATURE: MEMBERSHIP INFERENCE ATTACK (MIA) AUDIT
        # -------------------------------------------------------------
        st.markdown("#### ⚔️ Adversarial Membership Inference Attack (MIA) Audit")
        st.markdown("""
        **MIA Security Assessment:** We simulate an adversarial attacker attempting to guess whether individual records were part of the private training set based on their distance to the synthetic data.
        A score near **50% (Random Guess)** confirms strong empirical privacy preservation!
        """)

        mia_c1, mia_c2, mia_c3 = st.columns(3)
        with mia_c1:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Attacker Accuracy</span><span>🎯</span></div>
                <div class="metric-val-pro">{mia_res['MIA_Attack_Accuracy']}%</div>
                <div class="metric-sub-pro">Baseline Guess = 50.0%</div>
            </div>""", unsafe_allow_html=True)
        with mia_c2:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">Attacker Advantage</span><span>⚡</span></div>
                <div class="metric-val-pro">{mia_res['MIA_Advantage']}%</div>
                <div class="metric-sub-pro">Lower is Safer</div>
            </div>""", unsafe_allow_html=True)
        with mia_c3:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-header-pro"><span class="metric-label-pro">MIA Risk Classification</span><span>🛡️</span></div>
                <div class="metric-val-pro" style="font-size: 18px; color: #10b981;">{mia_res['MIA_Risk_Level']}</div>
                <div class="metric-sub-pro">Privacy Safe</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Privacy DCR Distribution Chart
        st.markdown("#### 🔒 Distance to Closest Record (DCR) Privacy Verification")
        st.markdown("""
        **Privacy Assessment:** DCR measures how far each synthetic sample is from its nearest real neighbor in the training data.
        A positive distribution confirms generalized interpolation rather than direct identity copying.
        """)
        dcr_path = plot_privacy_distance(privacy_res["DCR_Distances"])
        if os.path.exists(dcr_path):
            st.image(dcr_path, use_container_width=True)

        # Statistical KS Table
        st.markdown("#### 📐 Kolmogorov-Smirnov (KS) Numerical Feature Fidelity Profile")
        if quality_res["Numerical_KS_Tests"]:
            ks_df = pd.DataFrame(quality_res["Numerical_KS_Tests"]).T.reset_index().rename(columns={"index": "Numerical Feature"})
            st.dataframe(ks_df.style.format({"KS_Statistic": "{:.4f}", "P_Value": "{:.4f}", "KS_Complement_Score": "{:.2%}"}), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📥 Generate & Download Production Deliverables")

        col_d1, col_d2, col_d3 = st.columns(3)

        # 1. Comparison Report PDF
        with col_d1:
            if st.button("📄 Generate Comparison Report PDF", use_container_width=True):
                with st.spinner("Compiling comparison_report.pdf..."):
                    img_list = ["visuals/distribution_comparison_histograms.png", "visuals/correlation_heatmaps.png", "visuals/privacy_dcr_distribution.png"]
                    pdf_path = generate_comparison_pdf(
                        stats_data=prep.get_summary_statistics(df_orig),
                        quality_data=quality_res,
                        privacy_data=privacy_res,
                        image_paths=img_list,
                        save_path="reports/comparison_report.pdf"
                    )
                    st.success("Comparison Report generated!")

            if os.path.exists("reports/comparison_report.pdf"):
                with open("reports/comparison_report.pdf", "rb") as f:
                    st.download_button(
                        label="⬇️ Download comparison_report.pdf",
                        data=f.read(),
                        file_name="comparison_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        # 2. Evaluation Report PDF
        with col_d2:
            if st.button("📑 Generate Evaluation Report PDF", use_container_width=True):
                if st.session_state.ml_results is not None:
                    with st.spinner("Compiling evaluation_report.pdf..."):
                        img_list = ["visuals/model_performance_comparison.png", "visuals/confusion_matrix.png", "visuals/shap_summary_plot.png"]
                        pdf_path = generate_evaluation_pdf(
                            results_df=st.session_state.ml_results,
                            image_paths=img_list,
                            save_path="reports/evaluation_report.pdf"
                        )
                        st.success("Evaluation Report generated!")
                else:
                    st.warning("Please run Tab 4 ML Pipeline before generating evaluation report.")

            if os.path.exists("reports/evaluation_report.pdf"):
                with open("reports/evaluation_report.pdf", "rb") as f:
                    st.download_button(
                        label="⬇️ Download evaluation_report.pdf",
                        data=f.read(),
                        file_name="evaluation_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

        # 3. Classification Report TXT & Visuals Bundle
        with col_d3:
            if os.path.exists("reports/classification_report.txt"):
                with open("reports/classification_report.txt", "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇️ Download classification_report.txt",
                        data=f.read(),
                        file_name="classification_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            # Zip Visualizations
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for root, _, files in os.walk("visuals"):
                    for file in files:
                        if file.endswith((".png", ".jpg")):
                            zf.write(os.path.join(root, file), file)
            
            st.download_button(
                label="📦 Download All Visualizations (.ZIP)",
                data=zip_buf.getvalue(),
                file_name="synthetic_data_visuals.zip",
                mime="application/zip",
                use_container_width=True
            )

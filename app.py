# Author: [Your Name] | PRN: [Your PRN] | MIT-WPU F.Y. B.Tech Python Mini Project
# Module: app.py - Interactive Streamlit Control Plane Dashboard

import streamlit as st
import pandas as pd
import plotly.express as px
from engine import sync_openrouter_pricing, redact_pii, select_route, simulate_telemetry
from analytics import generate_baseline_csv, preprocess_and_score, build_statistical_figures

st.set_page_config(page_title="Enterprise LLM Control Plane", layout="wide")

# Initialize session state variables
if "models" not in st.session_state:
    models, is_live = sync_openrouter_pricing()
    st.session_state.models = models
    st.session_state.is_live = is_live

if "telemetry_df" not in st.session_state:
    csv_path = generate_baseline_csv()
    raw_df = pd.read_csv(csv_path)
    st.session_state.telemetry_df = preprocess_and_score(raw_df)

if "audit_logs" not in st.session_state:
    st.session_state.audit_logs = ["[SYSTEM] Control Plane initialized. Baseline telemetry loaded."]

# Header and Live Sync Status
st.title("Enterprise LLM Price-Performance Intelligence Control Plane")
if st.session_state.is_live:
    st.success("STATUS: Live Global Pricing Synced (OpenRouter Public Catalog Active)")
else:
    st.warning("STATUS: Offline Fallback Mode Active (Using Local Enterprise Tariff Profiles)")

# Sidebar Controls
st.sidebar.header("Control Plane Configuration")
if st.sidebar.button("Re-Sync Global Tariffs"):
    models, is_live = sync_openrouter_pricing()
    st.session_state.models = models
    st.session_state.is_live = is_live
    st.session_state.audit_logs.append(f"[SYNC] Tariff catalog refreshed (Live={is_live}).")

policy = st.sidebar.radio(
    "Algorithmic Routing Policy",
    ["Broadcast All", "Pareto Optimal", "Lowest Cost", "Lowest Latency"],
)
task_type = st.sidebar.selectbox(
    "Workload Category",
    ["Summarization", "Code Generation", "Classification", "Reasoning"],
)
enable_pii = st.sidebar.checkbox("Enable Responsible AI PII Redaction", value=True)

st.sidebar.divider()
st.sidebar.subheader("Historical CSV Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Telemetry CSV", type=["csv"])
if uploaded_file is not None:
    try:
        new_df = pd.read_csv(uploaded_file)
        cleaned_new = preprocess_and_score(new_df)
        st.session_state.telemetry_df = pd.concat(
            [st.session_state.telemetry_df, cleaned_new], ignore_index=True
        )
        st.session_state.audit_logs.append(f"[IMPORT] Ingested and normalized {len(cleaned_new)} rows.")
        st.sidebar.success(f"Loaded {len(cleaned_new)} rows.")
    except Exception as e:
        st.sidebar.error(f"CSV Error: {e}")

# Main Prompt Dispatch Section
st.subheader("1. Workload Dispatch & Responsible AI Gateway")
prompt_input = st.text_area(
    "Enter Enterprise Prompt:",
    value="Summarize Q3 financial compliance notes for client rahul.sharma@enterprise.in (Phone: 9876543210).",
    height=90,
)

if st.button("Dispatch Workload", type="primary"):
    processed_prompt = prompt_input
    if enable_pii:
        processed_prompt, redacted_count = redact_pii(prompt_input)
        if redacted_count > 0:
            st.info(f"Responsible AI Filter masked {redacted_count} PII item(s): `{processed_prompt}`")
            st.session_state.audit_logs.append(f"[PRIVACY] Redacted {redacted_count} PII entities before routing.")

    chosen_models = select_route(st.session_state.models, policy)
    new_records, new_logs = simulate_telemetry(processed_prompt, chosen_models, task_type)

    new_df = preprocess_and_score(pd.DataFrame(new_records))
    st.session_state.telemetry_df = pd.concat(
        [st.session_state.telemetry_df, new_df], ignore_index=True
    )
    st.session_state.audit_logs.extend(new_logs)

# KPI Metrics Row
df = st.session_state.telemetry_df
avg_by_model = df.groupby(["provider", "model"], as_index=False).agg(
    avg_tps=("tps", "mean"),
    avg_cost=("cost_per_1k", "mean"),
    avg_ttft=("ttft", "mean"),
    avg_pareto=("pareto_score", "mean"),
)

best_row = avg_by_model.loc[avg_by_model["avg_pareto"].idxmax()]
max_cost_val = avg_by_model["avg_cost"].max()
savings_pct = round(((max_cost_val - best_row["avg_cost"]) / max_cost_val) * 100, 1) if max_cost_val > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Top Pareto Model", best_row["model"].split("/")[-1])
col2.metric("Optimal Cost / 1K Calls", f"${best_row['avg_cost']:.4f}", f"-{savings_pct}% vs Max")
col3.metric("Optimal Avg TTFT", f"{best_row['avg_ttft']:.3f} s")
col4.metric("Total Telemetry Runs", len(df))

# Tabs for Dashboard, Statistical EDA, and Audit Logs
tab1, tab2, tab3 = st.tabs(["Live Price-Performance Charts", "Pandas & Seaborn EDA", "Audit Terminal & Export"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig_tps = px.bar(
            avg_by_model,
            x="provider",
            y="avg_tps",
            color="provider",
            title="Throughput Velocity (Tokens/Sec - Higher is Better)",
            labels={"avg_tps": "Tokens / Sec", "provider": "AI Provider"},
        )
        st.plotly_chart(fig_tps, use_container_width=True)

    with c2:
        fig_cost = px.bar(
            avg_by_model,
            x="provider",
            y="avg_cost",
            color="provider",
            title="Financial Footprint (Cost per 1K Calls $ - Lower is Better)",
            labels={"avg_cost": "USD per 1,000 Calls", "provider": "AI Provider"},
        )
        st.plotly_chart(fig_cost, use_container_width=True)

with tab2:
    st.markdown("### Preprocessed & Min-Max Normalized Telemetry Dataset")
    st.dataframe(df.tail(15), use_container_width=True)
    st.markdown("### Statistical Distribution & Correlation Analysis (Matplotlib + Seaborn)")
    stat_fig = build_statistical_figures(df)
    st.pyplot(stat_fig)

with tab3:
    st.markdown("### Real-Time Routing & Telemetry Audit Console")
    st.code("\n".join(st.session_state.audit_logs[-15:]), language="text")
    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Full Telemetry Audit CSV",
        data=csv_export,
        file_name="enterprise_llm_audit.csv",
        mime="text/csv",
    )

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REQUIRED_COLS = ["provider", "model", "task_type", "tokens", "ttft", "tps", "cost_per_1k"]


def generate_baseline_csv(filename="historical_telemetry.csv", rows=48):
    """Creates a realistic 48-row telemetry CSV dataset with intentional nulls for cleaning."""
    if os.path.exists(filename):
        return filename

    models = [
        ("OpenAI", "openai/gpt-4o", 0.65, 75.0, 1.85),
        ("Anthropic", "anthropic/claude-3.5-sonnet", 0.72, 68.0, 2.40),
        ("Google", "google/gemini-2.5-flash", 0.28, 145.0, 0.11),
        ("Meta (Groq)", "meta-llama/llama-3.3-70b-instruct", 0.18, 260.0, 0.21),
        ("DeepSeek", "deepseek/deepseek-chat", 0.45, 90.0, 0.06),
        ("Mistral", "mistralai/mistral-small", 0.32, 120.0, 0.12),
    ]
    tasks = ["Summarization", "Code Generation", "Classification", "Reasoning"]
    records = []

    for i in range(rows):
        prov, mod, base_ttft, base_tps, base_cost = models[i % len(models)]
        jitter = random.uniform(0.85, 1.20)
        records.append({
            "timestamp": (datetime.now() - timedelta(minutes=rows - i)).strftime("%Y-%m-%d %H:%M:%S"),
            "provider": prov,
            "model": mod,
            "task_type": random.choice(tasks),
            "tokens": random.randint(150, 600),
            "ttft": round(base_ttft * jitter, 3) if i not in (4, 17) else np.nan,
            "tps": round(base_tps / jitter, 1),
            "cost_per_1k": round(base_cost * jitter, 4),
        })

    pd.DataFrame(records).to_csv(filename, index=False)
    return filename


def preprocess_and_score(df: pd.DataFrame):
    """Validates columns, removes duplicates, fills nulls, and computes NumPy Pareto score."""
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")

    clean_df = df.drop_duplicates().copy()

    for col in ["tokens", "ttft", "tps", "cost_per_1k"]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # NumPy vectorized Min-Max normalization
    cost_arr = clean_df["cost_per_1k"].to_numpy(dtype=float)
    ttft_arr = clean_df["ttft"].to_numpy(dtype=float)

    cost_denom = np.max(cost_arr) - np.min(cost_arr)
    ttft_denom = np.max(ttft_arr) - np.min(ttft_arr)

    norm_cost = np.where(cost_denom == 0, 0.0, (cost_arr - np.min(cost_arr)) / cost_denom)
    norm_ttft = np.where(ttft_denom == 0, 0.0, (ttft_arr - np.min(ttft_arr)) / ttft_denom)

    clean_df["norm_cost"] = np.round(norm_cost, 3)
    clean_df["norm_ttft"] = np.round(norm_ttft, 3)
    clean_df["pareto_score"] = np.round((1.0 - (0.5 * norm_cost + 0.5 * norm_ttft)) * 100, 1)

    return clean_df


def build_statistical_figures(df: pd.DataFrame):
    """Creates Matplotlib + Seaborn statistical EDA plots."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.boxplot(data=df, x="provider", y="ttft", ax=axes[0], palette="viridis")
    axes[0].set_title("Latency (TTFT) Distribution by Provider")
    axes[0].set_ylabel("Time to First Token (sec)")
    axes[0].tick_params(axis="x", rotation=25)

    corr_cols = ["tokens", "ttft", "tps", "cost_per_1k", "pareto_score"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1])
    axes[1].set_title("Telemetry Metric Correlation Matrix")

    plt.tight_layout()
    return fig

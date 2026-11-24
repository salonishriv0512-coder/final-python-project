import urllib.request
import json
import re
import random
from datetime import datetime

# Built-in fallback catalog if lab Wi-Fi blocks external requests
FALLBACK_MODELS = [
    {"id": "openai/gpt-4o", "provider": "OpenAI", "input_cost_per_m": 2.50, "output_cost_per_m": 10.00, "base_ttft": 0.65, "base_tps": 75.0},
    {"id": "anthropic/claude-3.5-sonnet", "provider": "Anthropic", "input_cost_per_m": 3.00, "output_cost_per_m": 15.00, "base_ttft": 0.72, "base_tps": 68.0},
    {"id": "google/gemini-2.5-flash", "provider": "Google", "input_cost_per_m": 0.15, "output_cost_per_m": 0.60, "base_ttft": 0.28, "base_tps": 145.0},
    {"id": "meta-llama/llama-3.3-70b-instruct", "provider": "Meta (Groq)", "input_cost_per_m": 0.59, "output_cost_per_m": 0.79, "base_ttft": 0.18, "base_tps": 260.0},
    {"id": "deepseek/deepseek-chat", "provider": "DeepSeek", "input_cost_per_m": 0.14, "output_cost_per_m": 0.28, "base_ttft": 0.45, "base_tps": 90.0},
    {"id": "mistralai/mistral-small", "provider": "Mistral", "input_cost_per_m": 0.20, "output_cost_per_m": 0.60, "base_ttft": 0.32, "base_tps": 120.0},
]


def sync_openrouter_pricing():
    """Fetches live model tariffs from OpenRouter public API with offline fallback."""
    url = "https://openrouter.ai"
    target_map = {m["id"]: m.copy() for m in FALLBACK_MODELS}

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            raw_data = json.loads(response.read().decode("utf-8"))

        synced = []
        for item in raw_data.get("data", []):
            model_id = item.get("id")
            if model_id in target_map:
                pricing = item.get("pricing", {})
                in_rate = float(pricing.get("prompt", 0)) * 1_000_000
                out_rate = float(pricing.get("completion", 0)) * 1_000_000
                entry = target_map[model_id]
                if in_rate > 0:
                    entry["input_cost_per_m"] = round(in_rate, 3)
                if out_rate > 0:
                    entry["output_cost_per_m"] = round(out_rate, 3)
                synced.append(entry)

        if len(synced) >= 3:
            return synced, True
        return FALLBACK_MODELS, False
    except Exception:
        return FALLBACK_MODELS, False


def redact_pii(prompt: str):
    """Responsible AI Guardrail: Masks emails and 10-digit phone numbers using regex."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"

    cleaned, email_count = re.subn(email_pattern, "[EMAIL_REDACTED]", prompt)
    cleaned, phone_count = re.subn(phone_pattern, "[PHONE_REDACTED]", cleaned)
    return cleaned, (email_count + phone_count)


def select_route(models: list, policy: str):
    """Selects optimal model(s) based on the chosen routing policy."""
    if policy == "Broadcast All":
        return models

    if policy == "Lowest Cost":
        best = min(models, key=lambda m: m["input_cost_per_m"] + m["output_cost_per_m"])
        return [best]

    if policy == "Lowest Latency":
        best = min(models, key=lambda m: m["base_ttft"])
        return [best]

    # Default: Pareto Optimal (50% Cost + 50% Latency normalized)
    max_cost = max(m["input_cost_per_m"] + m["output_cost_per_m"] for m in models) or 1.0
    max_ttft = max(m["base_ttft"] for m in models) or 1.0

    best = min(
        models,
        key=lambda m: 0.5 * ((m["input_cost_per_m"] + m["output_cost_per_m"]) / max_cost)
        + 0.5 * (m["base_ttft"] / max_ttft),
    )
    return [best]


def simulate_telemetry(prompt: str, selected_models: list, task_type: str = "General"):
    """Simulates execution metrics: tokens, TTFT, throughput (TPS), and cost per 1K calls."""
    results = []
    logs = []
    input_tokens = max(12, int(len(prompt.split()) * 1.35))

    for m in selected_models:
        output_tokens = random.randint(120, 320)
        jitter = random.uniform(0.88, 1.18)

        ttft = round(m["base_ttft"] * jitter, 3)
        tps = round(m["base_tps"] / jitter, 1)
        single_call_cost = (
            input_tokens * m["input_cost_per_m"] + output_tokens * m["output_cost_per_m"]
        ) / 1_000_000
        cost_per_1k = round(single_call_cost * 1000, 4)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results.append({
            "timestamp": now_str,
            "provider": m["provider"],
            "model": m["id"],
            "task_type": task_type,
            "tokens": input_tokens + output_tokens,
            "ttft": ttft,
            "tps": tps,
            "cost_per_1k": cost_per_1k,
        })

        logs.append(
            f"[{now_str}] ROUTED -> {m['id']} | Tokens: {input_tokens + output_tokens} "
            f"| TTFT: {ttft}s | TPS: {tps} tok/s | Cost/1K: ${cost_per_1k}"
        )

    return results, logs

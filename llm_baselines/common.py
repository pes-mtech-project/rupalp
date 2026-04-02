import json
import pickle
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

ACTION_MAP = {"sell": -1, "hold": 0, "buy": 1}


def load_subset_frame(subset_pkl_path: str, ticker: str) -> pd.DataFrame:
    with open(subset_pkl_path, "rb") as f:
        env = pickle.load(f)

    rows: List[Dict[str, Any]] = []
    for dt in sorted(env.keys()):
        price = env[dt].get("price", {}).get(ticker)
        if price is None:
            continue
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "close": float(price),
                "news": env[dt].get("news", {}).get(ticker, []),
                "filing_k": env[dt].get("filing_k", {}).get(ticker, ""),
                "filing_q": env[dt].get("filing_q", {}).get(ticker, ""),
            }
        )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"Need at least 2 price rows for {ticker}.")
    return df


def split_frame(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    out = df.loc[mask].reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("Selected split is too small.")
    return out


def build_llm_client(
    model: str,
    end_point: str,
    system_message: str,
    temperature: float = 0.0,
):
    api_key = os.environ.get("OPENAI_API_KEY", "-")
    is_ollama_native_chat = end_point.rstrip("/").endswith("/api/chat")
    is_openai_chat = end_point.rstrip("/").endswith("/v1/chat/completions")

    def _call(prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        if is_ollama_native_chat:
            response = httpx.post(
                end_point,
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=600.0,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

        if is_openai_chat:
            response = httpx.post(
                end_point,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout=600.0,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        raise ValueError(f"Unsupported baseline endpoint: {end_point}")

    return _call


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def parse_decision(raw_text: str) -> Tuple[str, str]:
    parsed = _extract_json_block(raw_text) or {}
    decision = str(parsed.get("decision", "")).strip().lower()
    summary = str(parsed.get("summary", "")).strip()

    if decision not in ACTION_MAP:
        lowered = raw_text.lower()
        for option in ("buy", "sell", "hold"):
            if option in lowered:
                decision = option
                break
    if decision not in ACTION_MAP:
        decision = "hold"
    if not summary:
        summary = raw_text.strip()[:500]
    return decision, summary


def truncate_news(news_items: List[str], max_items: int = 8, max_chars: int = 350) -> List[str]:
    out = []
    for item in news_items[:max_items]:
        clean = " ".join(str(item).split())
        out.append(clean[:max_chars])
    return out


def recent_return_summary(window_df: pd.DataFrame) -> str:
    rows = []
    for idx in range(len(window_df)):
        date_str = window_df.loc[idx, "date"].date().isoformat()
        close = float(window_df.loc[idx, "close"])
        if idx == 0:
            ret_1 = 0.0
        else:
            prev_close = float(window_df.loc[idx - 1, "close"])
            ret_1 = float(np.log(close / prev_close))
        rows.append(f"{date_str}: close={close:.4f}, log_ret_1d={ret_1:.6f}")
    return "\n".join(rows)


def compute_metrics(result_df: pd.DataFrame) -> Dict[str, float]:
    rewards = result_df["reward"].to_numpy(dtype=float)
    cumulative_return = float(rewards.sum())
    daily_std = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    annualized_vol = float(daily_std * np.sqrt(252))
    sharpe = float("nan") if annualized_vol == 0 else float((cumulative_return / (len(result_df) / 252)) / annualized_vol)
    wealth = [1.0]
    for r in rewards:
        wealth.append(wealth[-1] * (1.0 + r))
    peak = wealth[0]
    max_drawdown = 0.0
    for v in wealth:
        if v > peak:
            peak = v
        max_drawdown = max(max_drawdown, (peak - v) / peak)
    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe,
        "daily_volatility": daily_std,
        "annualized_volatility": annualized_vol,
        "max_drawdown": float(max_drawdown),
    }


def load_resume_state(actions_csv: Path) -> Tuple[List[Dict[str, Any]], int, float]:
    if not actions_csv.exists():
        return [], 0, 0.0
    existing = pd.read_csv(actions_csv)
    if existing.empty:
        return [], 0, 0.0
    rows = existing.to_dict(orient="records")
    return rows, len(rows), float(existing.iloc[-1]["cumulative_return"])

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root is importable for unpickling puppy classes.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_models(model_args: List[str]) -> List[Tuple[str, str]]:
    models: List[Tuple[str, str]] = []
    for item in model_args:
        if "=" not in item:
            raise ValueError(f"Invalid --model '{item}'. Use NAME=PATH.")
        name, path = item.split("=", 1)
        models.append((name.strip(), path.strip()))
    return models


def _load_prices(env_pkl: str, ticker: str, start: str, end: str) -> pd.Series:
    with open(env_pkl, "rb") as f:
        env = pickle.load(f)
    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date()
    rows = []
    for d in sorted(env.keys()):
        if d < start_dt or d > end_dt:
            continue
        price = env[d].get("price", {}).get(ticker)
        if price is not None:
            rows.append((d, float(price)))
    if not rows:
        raise ValueError("No price rows found in selected date range.")
    s = pd.Series({d: p for d, p in rows}).sort_index()
    return s


def _load_actions_from_state_dict(state_dict_pkl: str, start: str, end: str) -> pd.Series:
    with open(state_dict_pkl, "rb") as f:
        state = pickle.load(f)
    action_series = state["portfolio"].action_series
    s = pd.Series(action_series).sort_index()
    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date()
    s = s[(s.index >= start_dt) & (s.index <= end_dt)].astype(float)
    if s.empty:
        raise ValueError(
            f"No actions found in [{start}, {end}] for state_dict: {state_dict_pkl}. "
            "Use a TEST output state_dict for test-window plotting."
        )
    return s


def _cumulative_reward_list(prices: pd.Series, actions: pd.Series) -> List[float]:
    dates = list(prices.index)
    px = prices.values
    out = [0.0]
    reward = 0.0
    for i in range(len(dates) - 1):
        act = float(actions.get(dates[i], 0.0))
        reward += act * np.log(px[i + 1] / px[i])
        out.append(float(reward))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize cumulative return for multiple model outputs."
    )
    parser.add_argument(
        "--env-pkl",
        default="data/06_input/subset_symbols_TSLA.pkl",
        help="Path to subset_symbols_<TICKER>.pkl used for simulation.",
    )
    parser.add_argument("--ticker", default="TSLA", help="Ticker symbol.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as NAME=PATH_TO_STATE_DICT_PKL (repeatable).",
    )
    parser.add_argument(
        "--output",
        default="data/07_model_output/tsla_cumulative_returns.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    models = _parse_models(args.model)
    prices = _load_prices(args.env_pkl, args.ticker, args.start_date, args.end_date)
    dates = pd.to_datetime(list(prices.index))

    labels = ["B_H"]
    returns = []

    bh_actions = pd.Series({d: 1.0 for d in prices.index[:-1]})
    returns.append(_cumulative_reward_list(prices, bh_actions))

    for name, path in models:
        actions = _load_actions_from_state_dict(path, args.start_date, args.end_date)
        returns.append(_cumulative_reward_list(prices, actions))
        labels.append(name)

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, y in enumerate(returns):
        ax.plot(dates, y, label=labels[i], linewidth=2.2)

    ax.set_title(args.ticker, fontsize=22)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Cumulative Return", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()

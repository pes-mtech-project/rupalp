import argparse
import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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
    if len(rows) < 2:
        raise ValueError("Need at least 2 price rows in selected date range.")
    return pd.Series({d: p for d, p in rows}).sort_index()


def _load_actions(state_dict_pkl: str, start: str, end: str) -> pd.Series:
    with open(state_dict_pkl, "rb") as f:
        state = pickle.load(f)
    action_series = state["portfolio"].action_series
    s = pd.Series(action_series).sort_index().astype(float)
    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date()
    sliced = s[(s.index >= start_dt) & (s.index <= end_dt)]
    if sliced.empty:
        raise ValueError(
            f"No actions found in [{start}, {end}] for state_dict: {state_dict_pkl}. "
            "Use a TEST output state_dict for test-window Wilcoxon."
        )
    return sliced


def _cumulative_rewards(prices: pd.Series, actions: pd.Series) -> np.ndarray:
    dates = list(prices.index)
    px = prices.values
    out = []
    reward = 0.0
    for i in range(len(dates) - 1):
        act = float(actions.get(dates[i], 0.0))
        reward += act * np.log(px[i + 1] / px[i])
        out.append(reward)
    return np.array(out, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise Wilcoxon test across model cumulative rewards.")
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
        default="data/07_model_output/tsla_wilcoxon.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    models = _parse_models(args.model)
    prices = _load_prices(args.env_pkl, args.ticker, args.start_date, args.end_date)

    rewards_by_model: Dict[str, np.ndarray] = {}
    for name, path in models:
        actions = _load_actions(path, args.start_date, args.end_date)
        rewards_by_model[name] = _cumulative_rewards(prices, actions)

    results = []
    for m1, m2 in combinations(rewards_by_model.keys(), 2):
        x = rewards_by_model[m1]
        y = rewards_by_model[m2]
        n = min(len(x), len(y))
        if n == 0:
            continue
        stat, pval = wilcoxon(x[:n], y[:n])
        results.append({"model_1": m1, "model_2": m2, "statistic": stat, "p_value": pval})
        print(f"{m1} vs {m2}: statistic={stat}, p_value={pval}")

    out_df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved Wilcoxon results to {out_path}")


if __name__ == "__main__":
    main()

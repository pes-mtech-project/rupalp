import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
            "Use a TEST output state_dict for test-window metrics."
        )
    return sliced


def _daily_rewards(prices: pd.Series, actions: pd.Series) -> np.ndarray:
    dates = list(prices.index)
    px = prices.values
    out = []
    for i in range(len(dates) - 1):
        act = float(actions.get(dates[i], 0.0))
        out.append(act * np.log(px[i + 1] / px[i]))
    return np.array(out, dtype=float)


def _std_dev(rewards: np.ndarray) -> float:
    if len(rewards) <= 1:
        return 0.0
    return float(np.std(rewards, ddof=1))


def _total_reward(rewards: np.ndarray) -> float:
    return float(np.sum(rewards))


def _ann_vol(daily_std: float, trading_days: int = 252) -> float:
    return float(daily_std * np.sqrt(trading_days))


def _sharpe(cum_return: float, ann_volatility: float, n_prices: int) -> float:
    if ann_volatility == 0:
        return float("nan")
    annualized_return = cum_return / (n_prices / 252)
    return float(annualized_return / ann_volatility)


def _max_drawdown(daily_returns: np.ndarray) -> float:
    cumulative = [1.0]
    for r in daily_returns:
        cumulative.append(cumulative[-1] * (1.0 + r))
    peak = cumulative[0]
    max_dd = 0.0
    for v in cumulative:
        if v > peak:
            peak = v
        drawdown = (peak - v) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    return float(max_dd)


def _calc_metrics(prices: pd.Series, actions: pd.Series) -> Tuple[float, float, float, float, float]:
    dr = _daily_rewards(prices, actions)
    std_dev = _std_dev(dr)
    ann = _ann_vol(std_dev)
    cum = _total_reward(dr)
    shp = _sharpe(cum, ann, len(prices))
    mdd = _max_drawdown(dr)
    return cum, shp, std_dev, ann, mdd


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate trading metrics from model state_dict outputs.")
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
        default="data/07_model_output/tsla_metrics.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    models = _parse_models(args.model)
    prices = _load_prices(args.env_pkl, args.ticker, args.start_date, args.end_date)

    rows: Dict[str, Tuple[float, float, float, float, float]] = {}
    bh_actions = pd.Series({d: 1.0 for d in prices.index[:-1]})
    rows["Buy & Hold"] = _calc_metrics(prices, bh_actions)

    for name, path in models:
        actions = _load_actions(path, args.start_date, args.end_date)
        rows[name] = _calc_metrics(prices, actions)

    out_df = pd.DataFrame(
        rows,
        index=[
            "Cumulative Return",
            "Sharpe Ratio",
            "Standard Deviation",
            "Annualized Volatility",
            "Max Drawdown",
        ],
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path)
    print(out_df)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()

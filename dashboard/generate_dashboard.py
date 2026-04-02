import argparse
import csv
import html
import math
import pickle
import sys
import types
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "data/07_model_output"
ASSET_ROOT = OUTPUT_ROOT / "dashboard_assets"

TEST_START = date(2026, 1, 2)
TEST_END = date(2026, 3, 6)
TICKERS = ["TSLA", "AMZN", "NFLX", "MSFT", "COIN"]

METRIC_COLUMNS = [
    "Cumulative Return (%)",
    "Sharpe Ratio",
    "Daily Volatility (%)",
    "Annualized Volatility (%)",
    "Max Drawdown (%)",
]

FINMEM_TEST_PATHS = {
    "TSLA": REPO_ROOT / "data/05_test_model_output/tsla_2025/agent_1/state_dict.pkl",
    "AMZN": REPO_ROOT / "data/05_test_model_output/amzn_2026/agent_1/state_dict.pkl",
    "NFLX": REPO_ROOT / "data/05_test_model_output/nflx_2026/agent_1/state_dict.pkl",
    "MSFT": REPO_ROOT / "data/05_test_model_output/msft_2026/agent_1/state_dict.pkl",
    "COIN": REPO_ROOT / "data/05_test_model_output/coin_2026/agent_1/state_dict.pkl",
}

RQ1_GPT35_SELF_ADAPTIVE_PATHS = {
    "TSLA": REPO_ROOT / "data/05_test_model_output/tsla_rq1_gpt35_self_adaptive_new/agent_1/state_dict.pkl",
    "AMZN": REPO_ROOT / "data/05_test_model_output/amzn_rq1_gpt35_self_adaptive_new/agent_1/state_dict.pkl",
    "NFLX": REPO_ROOT / "data/05_test_model_output/nflx_rq1_gpt35_self_adaptive_new/agent_1/state_dict.pkl",
    "MSFT": REPO_ROOT / "data/05_test_model_output/msft_rq1_gpt35_self_adaptive_new/agent_1/state_dict.pkl",
    "COIN": REPO_ROOT / "data/05_test_model_output/coin_rq1_gpt35_self_adaptive_new/agent_1/state_dict.pkl",
}
RQ1_GPT35_SELF_ADAPTIVE_FALLBACK_PATHS = {
    "TSLA": REPO_ROOT / "data/05_test_model_output/tsla_rq1_gpt35_self_adaptive/agent_1/state_dict.pkl",
    "AMZN": REPO_ROOT / "data/05_test_model_output/amzn_rq1_gpt35_self_adaptive/agent_1/state_dict.pkl",
    "NFLX": REPO_ROOT / "data/05_test_model_output/nflx_rq1_gpt35_self_adaptive/agent_1/state_dict.pkl",
    "MSFT": REPO_ROOT / "data/05_test_model_output/msft_rq1_gpt35_self_adaptive/agent_1/state_dict.pkl",
    "COIN": REPO_ROOT / "data/05_test_model_output/coin_rq1_gpt35_self_adaptive/agent_1/state_dict.pkl",
}

GA_ACTION_PATHS = {
    ticker: REPO_ROOT / f"data/llm_baselines/ga/{ticker}/{ticker}_ga_actions.csv"
    for ticker in TICKERS
}

DRL_ACTION_PATHS = {
    ticker: {
        "A2C": REPO_ROOT / f"data/drl_results/{ticker.lower()}_a2c/{ticker}_a2c_actions.csv",
        "PPO": REPO_ROOT / f"data/drl_results/{ticker.lower()}_ppo/{ticker}_ppo_actions.csv",
        "DQN": REPO_ROOT / f"data/drl_results/{ticker.lower()}_dqn/{ticker}_dqn_actions.csv",
    }
    for ticker in TICKERS
}

TSLA_TRADING_AGENT_PATHS = {
    "GPT 3.5-Turbo": REPO_ROOT / "data/05_test_model_output/tsla_2025/agent_1/state_dict.pkl",
    "GPT4": REPO_ROOT / "data/05_test_model_output/tsla_gpt4/agent_1/state_dict.pkl",
    "GPT4-Turbo": REPO_ROOT / "data/05_test_model_output/tsla_gpt4_turbo/agent_1/state_dict.pkl",
    "davinci-003": REPO_ROOT / "data/05_test_model_output/tsla_davinci_003_paper/agent_1/state_dict.pkl",
    "Llama2-70b-chat": REPO_ROOT / "data/05_test_model_output/tsla_llama2_70b_chat/agent_1/state_dict.pkl",
}

TSLA_LIMITED_TRAIN_PATH = REPO_ROOT / "data/05_test_model_output/tsla_split/agent_1/state_dict.pkl"
TSLA_TOPK_PATHS = {
    "Top 1": REPO_ROOT / "data/05_test_model_output/tsla_top1/agent_1/state_dict.pkl",
    "Top 3": REPO_ROOT / "data/05_test_model_output/tsla_top3/agent_1/state_dict.pkl",
    "Top 5": REPO_ROOT / "data/05_test_model_output/tsla_top5/agent_1/state_dict.pkl",
    "Top 10": REPO_ROOT / "data/05_test_model_output/tsla_top10/agent_1/state_dict.pkl",
}
TSLA_RISK_PATHS = {
    "Risk Averse": REPO_ROOT / "data/05_test_model_output/tsla_rq1_risk_averse/agent_1/state_dict.pkl",
    "Risk Seeking": REPO_ROOT / "data/05_test_model_output/tsla_rq1_risk_seeking/agent_1/state_dict.pkl",
    "Self-Adaptive": REPO_ROOT / "data/05_test_model_output/tsla_rq1_self_adaptive/agent_1/state_dict.pkl",
}
TSLA_GPT35_RISK_PATHS = {
    "Risk Averse": REPO_ROOT / "data/05_test_model_output/tsla_rq1_gpt35_risk_averse/agent_1/state_dict.pkl",
    "Risk Seeking": REPO_ROOT / "data/05_test_model_output/tsla_rq1_gpt35_risk_seeking/agent_1/state_dict.pkl",
    "Self-Adaptive": REPO_ROOT / "data/05_test_model_output/tsla_rq1_gpt35_self_adaptive/agent_1/state_dict.pkl",
}


@dataclass
class MethodResult:
    method: str
    status: str
    note: str
    metrics: Optional[Dict[str, float]]
    curve: Optional[List[float]]


def _subset_pkl_for_ticker(ticker: str) -> Path:
    return REPO_ROOT / f"data/06_input/subset_symbols_{ticker}.pkl"


def _install_pickle_stubs() -> None:
    if "puppy.portfolio" in sys.modules:
        return
    puppy_mod = types.ModuleType("puppy")
    portfolio_mod = types.ModuleType("puppy.portfolio")

    class Portfolio:
        pass

    portfolio_mod.Portfolio = Portfolio
    puppy_mod.portfolio = portfolio_mod
    sys.modules["puppy"] = puppy_mod
    sys.modules["puppy.portfolio"] = portfolio_mod


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _fmt_number(value: Optional[float]) -> str:
    if value is None:
        return "PLACEHOLDER"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NaN"
    return f"{value:.4f}"


def _fmt_metric_value(metric: str, value: Optional[float]) -> str:
    if value is None:
        return "PLACEHOLDER"
    return _fmt_number(value)


def _load_prices(ticker: str, start: date, end: date) -> Dict[date, float]:
    with open(_subset_pkl_for_ticker(ticker), "rb") as f:
        env = pickle.load(f)

    prices: Dict[date, float] = {}
    for cur_date in sorted(env.keys()):
        if cur_date < start or cur_date > end:
            continue
        price = env[cur_date].get("price", {}).get(ticker)
        if price is not None:
            prices[cur_date] = float(price)
    if len(prices) < 2:
        raise ValueError(f"Need at least two prices for {ticker} in selected window.")
    return prices


def _load_state_actions(path: Path, start: date, end: date) -> Dict[date, float]:
    _install_pickle_stubs()
    with open(path, "rb") as f:
        state = pickle.load(f)
    action_series = state["portfolio"].action_series
    out: Dict[date, float] = {}
    for cur_date, action in action_series.items():
        if start <= cur_date <= end:
            out[cur_date] = float(action)
    return out


def _load_action_csv(path: Path, start: date, end: date) -> Dict[date, float]:
    out: Dict[date, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur_date = _parse_date(row["date"])
            if start <= cur_date <= end:
                out[cur_date] = float(row["action"])
    return out


def _buy_hold_actions(prices: Dict[date, float]) -> Dict[date, float]:
    dates = sorted(prices.keys())
    return {cur_date: 1.0 for cur_date in dates[:-1]}


def _daily_rewards(prices: Dict[date, float], actions: Dict[date, float]) -> List[float]:
    dates = sorted(prices.keys())
    rewards: List[float] = []
    for idx in range(len(dates) - 1):
        cur_date = dates[idx]
        cur_price = prices[cur_date]
        next_price = prices[dates[idx + 1]]
        action = float(actions.get(cur_date, 0.0))
        rewards.append(action * math.log(next_price / cur_price))
    return rewards


def _cumulative_curve(prices: Dict[date, float], actions: Dict[date, float]) -> List[float]:
    dates = sorted(prices.keys())
    curve = [0.0]
    running = 0.0
    for idx in range(len(dates) - 1):
        cur_date = dates[idx]
        cur_price = prices[cur_date]
        next_price = prices[dates[idx + 1]]
        action = float(actions.get(cur_date, 0.0))
        running += action * math.log(next_price / cur_price)
        curve.append(running)
    return curve


def _max_drawdown(rewards: List[float]) -> float:
    cumulative = [1.0]
    for reward in rewards:
        cumulative.append(cumulative[-1] * (1.0 + reward))
    peak = cumulative[0]
    max_dd = 0.0
    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _compute_metrics(prices: Dict[date, float], actions: Dict[date, float]) -> Dict[str, float]:
    rewards = _daily_rewards(prices, actions)
    cum = float(sum(rewards))
    std = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    ann = float(std * math.sqrt(252))
    sharpe = float("nan")
    if ann != 0:
        annualized_return = cum / (len(prices) / 252)
        sharpe = float(annualized_return / ann)
    mdd = _max_drawdown(rewards)
    return {
        "Cumulative Return (%)": cum * 100.0,
        "Sharpe Ratio": sharpe,
        "Daily Volatility (%)": std * 100.0,
        "Annualized Volatility (%)": ann * 100.0,
        "Max Drawdown (%)": mdd * 100.0,
    }


def _load_method_from_state(method: str, prices: Dict[date, float], path: Path) -> MethodResult:
    if not path.exists():
        return MethodResult(method, "placeholder", f"Missing output: {path}", None, None)
    try:
        actions = _load_state_actions(path, TEST_START, TEST_END)
        if not actions:
            return MethodResult(method, "placeholder", f"No test actions in {path}", None, None)
        return MethodResult(
            method,
            "available",
            "",
            _compute_metrics(prices, actions),
            _cumulative_curve(prices, actions),
        )
    except Exception as exc:
        return MethodResult(method, "placeholder", f"Could not load {path}: {exc}", None, None)


def _load_method_from_csv(method: str, prices: Dict[date, float], path: Path) -> MethodResult:
    if not path.exists():
        return MethodResult(method, "placeholder", f"Missing output: {path}", None, None)
    try:
        actions = _load_action_csv(path, TEST_START, TEST_END)
        if not actions:
            return MethodResult(method, "placeholder", f"No test actions in {path}", None, None)
        return MethodResult(
            method,
            "available",
            "",
            _compute_metrics(prices, actions),
            _cumulative_curve(prices, actions),
        )
    except Exception as exc:
        return MethodResult(method, "placeholder", f"Could not load {path}: {exc}", None, None)


def _placeholder(method: str, note: str) -> MethodResult:
    return MethodResult(method, "placeholder", note, None, None)


def _pick_best_result(results: List[MethodResult]) -> Optional[MethodResult]:
    available = [result for result in results if result.metrics is not None]
    if not available:
        return None

    def key(result: MethodResult) -> Tuple[float, float, float]:
        metrics = result.metrics or {}
        return (
            float(metrics.get("Cumulative Return (%)", float("-inf"))),
            float(metrics.get("Sharpe Ratio", float("-inf"))),
            -float(metrics.get("Max Drawdown (%)", float("inf"))),
        )

    return max(available, key=key)


def _write_csv(path: Path, headers: List[str], rows: Iterable[Iterable[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _svg_text(text: str) -> str:
    return html.escape(text, quote=True)


def _write_table_svg(path: Path, title: str, headers: List[str], rows: List[List[str]]) -> None:
    col_widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    cell_widths = [max(120, width * 8 + 24) for width in col_widths]
    row_height = 28
    title_height = 44
    width = sum(cell_widths) + 40
    height = title_height + row_height * (len(rows) + 1) + 20

    x_positions = [20]
    for width_value in cell_widths[:-1]:
        x_positions.append(x_positions[-1] + width_value)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #1f2937; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".head { font-size: 12px; font-weight: 700; }",
        ".cell { font-size: 12px; }",
        "</style>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#fffdf8' stroke='#d4cfc7'/>",
        f"<text class='title' x='20' y='28'>{_svg_text(title)}</text>",
    ]

    y = title_height
    parts.append(f"<rect x='20' y='{y}' width='{sum(cell_widths)}' height='{row_height}' fill='#efe7db' stroke='#c7beb3'/>")
    for idx, header in enumerate(headers):
        parts.append(
            f"<text class='head' x='{x_positions[idx] + 8}' y='{y + 18}'>{_svg_text(header)}</text>"
        )

    for row_idx, row in enumerate(rows):
        row_y = y + row_height * (row_idx + 1)
        fill = "#ffffff" if row_idx % 2 == 0 else "#f8f5ef"
        parts.append(
            f"<rect x='20' y='{row_y}' width='{sum(cell_widths)}' height='{row_height}' fill='{fill}' stroke='#ddd6cc'/>"
        )
        for col_idx, value in enumerate(row):
            parts.append(
                f"<text class='cell' x='{x_positions[col_idx] + 8}' y='{row_y + 18}'>{_svg_text(value)}</text>"
            )

    current_x = 20
    for width_value in cell_widths:
        parts.append(
            f"<line x1='{current_x}' y1='{y}' x2='{current_x}' y2='{height - 12}' stroke='#d4cfc7' stroke-width='1'/>"
        )
        current_x += width_value
    parts.append(
        f"<line x1='{20 + sum(cell_widths)}' y1='{y}' x2='{20 + sum(cell_widths)}' y2='{height - 12}' stroke='#d4cfc7' stroke-width='1'/>"
    )
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def _write_line_chart_svg(
    path: Path,
    title: str,
    dates: List[date],
    series_map: Dict[str, List[float]],
    note: str = "",
) -> None:
    width = 1100
    height = 520
    margin_left = 90
    margin_right = 20
    margin_top = 60
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = [value for series in series_map.values() for value in series]
    min_y = min(all_values)
    max_y = max(all_values)
    if min_y == max_y:
        min_y -= 0.1
        max_y += 0.1
    padding = (max_y - min_y) * 0.1
    min_y -= padding
    max_y += padding

    style_map = {
        "Buy & Hold": ("#111111", "8 4 1.5 4"),
        "B&H": ("#111111", "8 4 1.5 4"),
        "FinMem": ("#d14749", ""),
        "FinMem Self-Adaptive (New)": ("#1f77b4", "10 4"),
        "FinGPT": ("#4e89e0", ""),
        "Generative Agents": ("#59a14f", ""),
        "GA": ("#59a14f", ""),
        "A2C": ("#ee4199", ""),
        "PPO": ("#f28e2b", ""),
        "DQN": ("#8F337F", ""),
        "GPT 3.5-Turbo": ("#59a14f", ""),
        "GPT4": ("#4e89e0", ""),
        "GPT4-Turbo": ("#d14749", ""),
        "davinci-003": ("#f28e2b", ""),
        "Llama2-70b-chat": ("#8F337F", ""),
        "Self-Adaptive": ("#d14749", ""),
        "Risk Seeking": ("#59a14f", ""),
        "Risk Averse": ("#4e89e0", ""),
        "FinMem_top1": ("#4e89e0", ""),
        "FinMem_top3": ("#59a14f", ""),
        "FinMem_top5": ("#d14749", ""),
        "FinMem_top10": ("#f28e2b", ""),
        "FinMem (Full Training)": ("#d14749", ""),
        "FinMem (Limited Training)": ("#4e89e0", ""),
    }
    fallback_colors = ["#4e89e0", "#d14749", "#59a14f", "#8F337F", "#f28e2b", "#8c564b"]
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #1f2937; }",
        ".title { font-size: 24px; font-weight: 700; }",
        ".axis { font-size: 12px; }",
        ".legend { font-size: 13px; }",
        "</style>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='#fffdf8' stroke='#d4cfc7'/>",
        f"<text class='title' x='{width / 2:.1f}' y='34' text-anchor='middle'>{_svg_text(title)}</text>",
    ]

    for tick_idx in range(6):
        value = min_y + (max_y - min_y) * tick_idx / 5.0
        y = margin_top + plot_height - (value - min_y) / (max_y - min_y) * plot_height
        parts.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' stroke='#e5dfd6' stroke-width='1'/>"
        )
        parts.append(
            f"<text class='axis' x='{margin_left - 10}' y='{y + 4:.2f}' text-anchor='end'>{value:.2f}</text>"
        )

    date_labels = [dates[0], dates[len(dates) // 2], dates[-1]]
    for label_date in date_labels:
        idx = dates.index(label_date)
        x = margin_left + (idx / (len(dates) - 1)) * plot_width
        parts.append(
            f"<line x1='{x:.2f}' y1='{margin_top}' x2='{x:.2f}' y2='{margin_top + plot_height}' stroke='#e5dfd6' stroke-width='1'/>"
        )
        parts.append(
            f"<text class='axis' x='{x:.2f}' y='{margin_top + plot_height + 22}' text-anchor='middle'>{label_date.isoformat()}</text>"
        )

    parts.append(
        f"<line x1='{margin_left}' y1='{margin_top + plot_height}' x2='{width - margin_right}' y2='{margin_top + plot_height}' stroke='#a8a29a' stroke-width='1.5'/>"
    )
    parts.append(
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' stroke='#a8a29a' stroke-width='1.5'/>"
    )

    for idx, (name, values) in enumerate(series_map.items()):
        color, dasharray = style_map.get(name, (fallback_colors[idx % len(fallback_colors)], ""))
        points = []
        for point_idx, value in enumerate(values):
            x = margin_left + (point_idx / (len(values) - 1)) * plot_width
            y = margin_top + plot_height - (value - min_y) / (max_y - min_y) * plot_height
            points.append(f"{x:.2f},{y:.2f}")
        dash_attr = f" stroke-dasharray='{dasharray}'" if dasharray else ""
        parts.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'{dash_attr} points='{' '.join(points)}'/>"
        )
        legend_y = margin_top + plot_height - 110 + idx * 20
        legend_x = margin_left + 16
        parts.append(
            f"<line x1='{legend_x}' y1='{legend_y}' x2='{legend_x + 24}' y2='{legend_y}' stroke='{color}' stroke-width='3' stroke-linecap='round'{dash_attr}/>"
        )
        parts.append(
            f"<text class='legend' x='{legend_x + 32}' y='{legend_y + 4}'>{_svg_text(name)}</text>"
        )

    if note:
        parts.append(
            f"<text class='axis' x='{margin_left}' y='{height - 20}'>{_svg_text(note)}</text>"
        )
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def _placeholder_curve(length: int) -> List[float]:
    return [0.0 for _ in range(length)]


def _comparison1_for_ticker(ticker: str) -> Tuple[List[MethodResult], Path]:
    prices = _load_prices(ticker, TEST_START, TEST_END)
    finmem_result = _load_method_from_state("FinMem", prices, FINMEM_TEST_PATHS[ticker])
    finmem_self_adaptive_new = _load_method_from_state(
        "FinMem Self-Adaptive (New)",
        prices,
        RQ1_GPT35_SELF_ADAPTIVE_PATHS[ticker],
    )
    if finmem_self_adaptive_new.metrics is None:
        fallback_result = _load_method_from_state(
            "FinMem Self-Adaptive (New)",
            prices,
            RQ1_GPT35_SELF_ADAPTIVE_FALLBACK_PATHS[ticker],
        )
        if fallback_result.metrics is not None:
            finmem_self_adaptive_new = MethodResult(
                "FinMem Self-Adaptive (New)",
                "available",
                "Using the existing self-adaptive GPT-3.5 run until the separate *_new RQ1 output is completed.",
                fallback_result.metrics,
                fallback_result.curve,
            )

    results = [
        MethodResult(
            "Buy & Hold",
            "available",
            "",
            _compute_metrics(prices, _buy_hold_actions(prices)),
            _cumulative_curve(prices, _buy_hold_actions(prices)),
        ),
        finmem_result,
        finmem_self_adaptive_new,
        _placeholder(
            "FinGPT",
            "FinGPT baseline is blocked. The required Hugging Face base model remains gated.",
        ),
        _load_method_from_csv("Generative Agents", prices, GA_ACTION_PATHS[ticker]),
        _load_method_from_csv("A2C", prices, DRL_ACTION_PATHS[ticker]["A2C"]),
        _load_method_from_csv("PPO", prices, DRL_ACTION_PATHS[ticker]["PPO"]),
        _load_method_from_csv("DQN", prices, DRL_ACTION_PATHS[ticker]["DQN"]),
    ]

    available = {result.method: result.curve for result in results if result.curve is not None}
    note_parts = [f"{result.method}: {result.note}" for result in results if result.status != "available"]
    chart_path = ASSET_ROOT / f"comparison_1_{ticker.lower()}_chart.svg"
    _write_line_chart_svg(
        chart_path,
        f"{ticker} - Buy & Hold vs FinMem / FinMem Self-Adaptive (New) / FinGPT / GA / A2C / PPO / DQN",
        sorted(prices.keys()),
        available,
        " | ".join(note_parts) if note_parts else "",
    )
    return results, chart_path


def _rows_for_results(results: List[MethodResult]) -> List[List[str]]:
    rows: List[List[str]] = []
    for result in results:
        metrics = result.metrics or {}
        rows.append(
            [
                result.method,
                result.status,
                result.note,
                _fmt_metric_value("Cumulative Return (%)", metrics.get("Cumulative Return (%)")),
                _fmt_metric_value("Sharpe Ratio", metrics.get("Sharpe Ratio")),
                _fmt_metric_value("Daily Volatility (%)", metrics.get("Daily Volatility (%)")),
                _fmt_metric_value(
                    "Annualized Volatility (%)", metrics.get("Annualized Volatility (%)")
                ),
                _fmt_metric_value("Max Drawdown (%)", metrics.get("Max Drawdown (%)")),
            ]
        )
    return rows


def _build_comparison1_outputs() -> List[Tuple[str, List[MethodResult], Path, Path, Path]]:
    all_results: List[Tuple[str, List[MethodResult], Path, Path, Path]] = []
    combined_csv_rows: List[List[str]] = []

    for ticker in TICKERS:
        results, chart_path = _comparison1_for_ticker(ticker)
        rows = _rows_for_results(results)
        csv_path = OUTPUT_ROOT / f"comparison_1_{ticker.lower()}_metrics.csv"
        _write_csv(csv_path, ["Method", "Status", "Note", *METRIC_COLUMNS], rows)
        svg_path = ASSET_ROOT / f"comparison_1_{ticker.lower()}_metrics.svg"
        _write_table_svg(
            svg_path,
            f"RQ1 - {ticker} Metrics",
            ["Method", "Status", *METRIC_COLUMNS],
            [row[:2] + row[3:] for row in rows],
        )
        all_results.append((ticker, results, chart_path, csv_path, svg_path))
        for row in rows:
            combined_csv_rows.append(
                [
                    ticker,
                    *row,
                ]
            )

    csv_path = OUTPUT_ROOT / "comparison_1_all_tickers_metrics.csv"
    _write_csv(
        csv_path,
        [
            "Ticker",
            "Method",
            "Status",
            "Note",
            *METRIC_COLUMNS,
        ],
        combined_csv_rows,
    )
    return all_results


def _build_tsla_trading_agents(prices: Dict[date, float]) -> Tuple[List[MethodResult], Path, Path, Path]:
    results = [
        MethodResult(
            "B&H",
            "available",
            "",
            _compute_metrics(prices, _buy_hold_actions(prices)),
            _cumulative_curve(prices, _buy_hold_actions(prices)),
        )
    ]
    for method, path in TSLA_TRADING_AGENT_PATHS.items():
        if method == "GPT4-Turbo":
            result = _load_method_from_state(method, prices, path)
            if result.metrics is None:
                risk_candidates = [
                    _load_method_from_state("Risk Averse", prices, TSLA_RISK_PATHS["Risk Averse"]),
                    _load_method_from_state("Risk Seeking", prices, TSLA_RISK_PATHS["Risk Seeking"]),
                    _load_method_from_state("Self-Adaptive", prices, TSLA_RISK_PATHS["Self-Adaptive"]),
                ]
                best_risk = _pick_best_result(risk_candidates)
                if best_risk is not None:
                    result = MethodResult(
                        "GPT4-Turbo",
                        "available",
                        (
                            "Fallback to the best completed GPT-4-Turbo risk-profile run "
                            f"({best_risk.method}) because the dedicated RQ3 GPT-4-Turbo run is missing."
                        ),
                        best_risk.metrics,
                        best_risk.curve,
                    )
            results.append(result)
            continue
        results.append(_load_method_from_state(method, prices, path))

    csv_rows = _rows_for_results(results)

    csv_path = OUTPUT_ROOT / "comparison_3_tsla_trading_agents_metrics.csv"
    _write_csv(
        csv_path,
        ["Method", "Status", "Note", *METRIC_COLUMNS],
        csv_rows,
    )
    svg_path = ASSET_ROOT / "comparison_3_tsla_trading_agents_metrics.svg"
    _write_table_svg(
        svg_path,
        "RQ3 - TSLA Trading Agents",
        ["Method", "Status", *METRIC_COLUMNS],
        [row[:2] + row[3:] for row in csv_rows],
    )
    available = {result.method: result.curve for result in results if result.curve is not None}
    note_parts = [f"{result.method}: {result.note}" for result in results if result.status != "available"]
    chart_path = ASSET_ROOT / "comparison_3_tsla_trading_agents_chart.svg"
    _write_line_chart_svg(
        chart_path,
        "TSLA - B&H vs GPT 3.5 / GPT4 / GPT4-Turbo / davinci-003 / Llama2-70b-chat",
        sorted(prices.keys()),
        available,
        " | ".join(note_parts) if note_parts else "",
    )
    return results, csv_path, svg_path, chart_path


def _build_tsla_limited_training(prices: Dict[date, float]) -> Tuple[Path, Path, Path]:
    full_result = _load_method_from_state("FinMem (Full Training)", prices, FINMEM_TEST_PATHS["TSLA"])
    limited_result = _load_method_from_state("FinMem (Limited Training)", prices, TSLA_LIMITED_TRAIN_PATH)

    rows = _rows_for_results(
        [
            MethodResult(
                "B&H",
                "available",
                "",
                _compute_metrics(prices, _buy_hold_actions(prices)),
                _cumulative_curve(prices, _buy_hold_actions(prices)),
            ),
            full_result,
            limited_result if limited_result.metrics is not None else _placeholder(
                "FinMem (Limited Training)",
                "No validated limited-training TSLA test output found. Existing tsla_split output is unavailable or unusable.",
            ),
        ]
    )

    csv_path = OUTPUT_ROOT / "comparison_2_tsla_limited_training_metrics.csv"
    _write_csv(csv_path, ["Method", "Status", "Note", *METRIC_COLUMNS], rows)
    svg_path = ASSET_ROOT / "comparison_2_tsla_limited_training_metrics.svg"
    _write_table_svg(
        svg_path,
        "RQ2 - TSLA Limited Training",
        ["Method", "Status", *METRIC_COLUMNS],
        [row[:2] + row[3:] for row in rows],
    )

    dates = sorted(prices.keys())
    curve_len = len(dates)
    series_map = {
        "B&H": _cumulative_curve(prices, _buy_hold_actions(prices)),
        "FinMem (Full Training)": full_result.curve if full_result.curve is not None else _placeholder_curve(curve_len),
        "FinMem (Limited Training)": limited_result.curve if limited_result.curve is not None else _placeholder_curve(curve_len),
    }
    note = (
        "Limited-training line uses current tsla_split output. Verify run provenance before treating it as paper-faithful RQ2."
        if limited_result.curve is not None
        else "Limited-training run is not yet available. Placeholder zero-line shown."
    )
    chart_path = ASSET_ROOT / "comparison_2_tsla_limited_training_chart.svg"
    _write_line_chart_svg(
        chart_path,
        "TSLA - Full Training vs Limited Training",
        dates,
        series_map,
        note,
    )
    return csv_path, svg_path, chart_path


def _build_tsla_risk_profiles(
    prices: Dict[date, float],
    risk_paths: Dict[str, Path],
    slug: str,
    title: str,
    pending_note: str,
) -> Tuple[Path, Path, Path, str]:
    risk_averse = _load_method_from_state("Risk Averse", prices, risk_paths["Risk Averse"])
    risk_seeking = _load_method_from_state("Risk Seeking", prices, risk_paths["Risk Seeking"])
    self_adaptive = _load_method_from_state("Self-Adaptive", prices, risk_paths["Self-Adaptive"])
    risk_results = [risk_averse, risk_seeking, self_adaptive]
    best_risk = _pick_best_result(risk_results)
    rows = _rows_for_results(
        [
            MethodResult(
                "B&H",
                "available",
                "",
                _compute_metrics(prices, _buy_hold_actions(prices)),
                _cumulative_curve(prices, _buy_hold_actions(prices)),
            ),
            *risk_results,
        ]
    )
    csv_path = OUTPUT_ROOT / f"comparison_3_tsla_risk_profiles_{slug}.csv"
    _write_csv(csv_path, ["Profile", "Status", "Note", *METRIC_COLUMNS], rows)
    svg_path = ASSET_ROOT / f"comparison_3_tsla_risk_profiles_{slug}.svg"
    _write_table_svg(
        svg_path,
        f"RQ4 - TSLA Risk Profiles ({title})",
        ["Profile", "Status", *METRIC_COLUMNS],
        [row[:2] + row[3:] for row in rows],
    )

    dates = sorted(prices.keys())
    curve_len = len(dates)
    series_map = {
        "B&H": _cumulative_curve(prices, _buy_hold_actions(prices)),
        "Self-Adaptive": self_adaptive.curve if self_adaptive.curve is not None else _placeholder_curve(curve_len),
        "Risk Seeking": risk_seeking.curve if risk_seeking.curve is not None else _placeholder_curve(curve_len),
        "Risk Averse": risk_averse.curve if risk_averse.curve is not None else _placeholder_curve(curve_len),
    }
    chart_path = ASSET_ROOT / f"comparison_3_tsla_risk_profiles_chart_{slug}.svg"
    _write_line_chart_svg(
        chart_path,
        f"TSLA - {title} Risk Profiles vs B&H",
        dates,
        series_map,
        (
            pending_note
            if any(result.curve is None for result in risk_results)
            else ""
        ),
    )
    conclusion = (
        f"Best TSLA risk profile so far for {title}: {best_risk.method}."
        if best_risk is not None
        else f"Best TSLA risk profile is pending for {title} because no dedicated run has completed yet."
    )
    return csv_path, svg_path, chart_path, conclusion


def _build_tsla_topk(prices: Dict[date, float]) -> Tuple[Path, Path, Path]:
    rows = []
    bh_metrics = _compute_metrics(prices, _buy_hold_actions(prices))
    rows.append(
        [
            "B&H",
            "available",
            "",
            *[_fmt_metric_value(metric, bh_metrics.get(metric)) for metric in METRIC_COLUMNS],
        ]
    )

    top_results: List[MethodResult] = []
    for method, path in TSLA_TOPK_PATHS.items():
        result = _load_method_from_state(method, prices, path)
        if method == "Top 3" and result.metrics is None:
            result = _load_method_from_state("Top 3", prices, REPO_ROOT / "data/05_test_model_output/tsla_2025/agent_1/state_dict.pkl")
            if result.metrics is not None:
                result = MethodResult(
                    "Top 3",
                    "available",
                    "Using current TSLA default FinMem run because top_k=3 in the default config.",
                    result.metrics,
                    result.curve,
                )
        top_results.append(result)

    for result in top_results:
        if result.metrics is None:
            rows.append([result.method, "placeholder", result.note or "No dedicated TSLA Top-K output found.", "PLACEHOLDER", "PLACEHOLDER", "PLACEHOLDER", "PLACEHOLDER", "PLACEHOLDER"])
        else:
            rows.append(
                [
                    result.method,
                    result.status,
                    result.note,
                    *[_fmt_metric_value(metric, result.metrics.get(metric)) for metric in METRIC_COLUMNS],
                ]
            )

    csv_path = OUTPUT_ROOT / "comparison_4_tsla_topk_metrics.csv"
    _write_csv(csv_path, ["Method", "Status", "Note", *METRIC_COLUMNS], rows)
    svg_path = ASSET_ROOT / "comparison_4_tsla_topk_metrics.svg"
    _write_table_svg(svg_path, "RQ5 - TSLA Top-K", ["Method", "Status", *METRIC_COLUMNS], [row[:2] + row[3:] for row in rows])
    dates = sorted(prices.keys())
    curve_len = len(dates)
    series_map = {
        "B&H": _cumulative_curve(prices, _buy_hold_actions(prices)),
        "FinMem_top1": top_results[0].curve if top_results[0].curve is not None else _placeholder_curve(curve_len),
        "FinMem_top3": top_results[1].curve if top_results[1].curve is not None else _placeholder_curve(curve_len),
        "FinMem_top5": top_results[2].curve if top_results[2].curve is not None else _placeholder_curve(curve_len),
        "FinMem_top10": top_results[3].curve if top_results[3].curve is not None else _placeholder_curve(curve_len),
    }
    chart_path = ASSET_ROOT / "comparison_4_tsla_topk_chart.svg"
    _write_line_chart_svg(
        chart_path,
        "TSLA - FinMem Top-K Comparison",
        dates,
        series_map,
        "",
    )
    return csv_path, svg_path, chart_path


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _build_html(
    comparison1: List[Tuple[str, List[MethodResult], Path, Path, Path]],
    comparison1_csv: Path,
    comparison2_csv: Path,
    comparison2_svg: Path,
    comparison2_chart: Path,
    comparison3_csv: Path,
    comparison3_svg: Path,
    comparison3_chart: Path,
    comparison3_risk_csv: Path,
    comparison3_risk_svg: Path,
    comparison3_risk_chart: Path,
    comparison3_risk_conclusion: str,
    comparison3_risk_gpt35_csv: Path,
    comparison3_risk_gpt35_svg: Path,
    comparison3_risk_gpt35_chart: Path,
    comparison3_risk_gpt35_conclusion: str,
    comparison4_csv: Path,
    comparison4_svg: Path,
    comparison4_chart: Path,
) -> str:
    ticker_cards = []
    for ticker, results, chart_path, csv_path, svg_path in comparison1:
        placeholder_notes = [result.note for result in results if result.status != "available"]
        note_html = "".join(f"<li>{html.escape(note)}</li>" for note in placeholder_notes)
        if not note_html:
            note_html = "<li>All requested methods for this ticker are available.</li>"
        ticker_cards.append(
            "<section class='card ticker-card'>"
            f"<div class='card-head'><h3>{ticker}</h3><span class='chip'>RQ1</span></div>"
            f"<button class='chart-button' type='button' data-fullsrc='../{_rel(chart_path)}' data-title='{ticker} comparison chart'>"
            f"<img class='zoomable' src='../{_rel(chart_path)}' alt='{ticker} comparison chart' />"
            "</button>"
            "<details class='details-block' open>"
            "<summary>Show Metrics Table</summary>"
            "<div class='table-shell'>"
            f"<button class='chart-button' type='button' data-fullsrc='../{_rel(svg_path)}' data-title='{ticker} metrics table'>"
            f"<img src='../{_rel(svg_path)}' alt='{ticker} metrics table' />"
            "</button>"
            f"<p><a href='../{_rel(csv_path)}'>Download {ticker} CSV</a></p>"
            "</div>"
            "</details>"
            "<details class='details-block'>"
            "<summary>Availability Notes</summary>"
            f"<ul>{note_html}</ul>"
            "</details>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FINMEM Dashboard</title>
  <style>
    :root {{
      --bg: #f5f1ea;
      --panel: #fffdf8;
      --panel-2: #faf6ef;
      --line: #ddd6cc;
      --line-strong: #c7beb3;
      --text: #1f2937;
      --muted: #5f6b7a;
      --accent: #7c2d12;
      --accent-2: #7e22ce;
      --shadow: 0 10px 30px rgba(31, 41, 55, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(124, 45, 18, 0.08), transparent 24%),
        radial-gradient(circle at top left, rgba(126, 34, 206, 0.06), transparent 18%),
        var(--bg);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    .hero, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    h1, h2, h3 {{
      margin-top: 0;
    }}
    h1 {{
      font-size: 2.2rem;
      margin-bottom: 10px;
    }}
    .subtle {{
      color: var(--muted);
      line-height: 1.55;
      max-width: 940px;
    }}
    .card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--accent);
      font-size: 0.8rem;
      font-weight: 700;
      white-space: nowrap;
    }}
    img {{
      width: 100%;
      border: 1px solid #e6dfd4;
      border-radius: 10px;
      background: white;
    }}
    .chart-button {{
      display: block;
      width: 100%;
      border: 0;
      padding: 0;
      margin: 0;
      background: transparent;
      cursor: zoom-in;
      text-align: inherit;
    }}
    .chart-button img {{
      transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .chart-button:hover img {{
      transform: translateY(-2px);
      box-shadow: 0 12px 24px rgba(31, 41, 55, 0.08);
    }}
    .links a {{
      display: inline-block;
      margin-right: 14px;
      margin-bottom: 10px;
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .section {{
      margin-top: 28px;
    }}
    .placeholder-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .placeholder {{
      background: #faf6ef;
      border: 1px dashed var(--line-strong);
      border-radius: 12px;
      padding: 14px;
    }}
    .details-block {{
      margin-top: 14px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .details-block summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--accent);
      list-style: none;
    }}
    .details-block summary::-webkit-details-marker {{
      display: none;
    }}
    .details-block[open] summary {{
      margin-bottom: 10px;
    }}
    .card p {{
      line-height: 1.55;
      color: var(--muted);
    }}
    ul {{
      margin-bottom: 0;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 18px;
    }}
    .table-shell {{
      overflow-x: auto;
    }}
    .section-label {{
      display: inline-block;
      margin-bottom: 10px;
      font-size: 0.86rem;
      font-weight: 700;
      color: var(--accent-2);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .modal {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(17, 24, 39, 0.82);
      z-index: 9999;
    }}
    .modal.open {{
      display: flex;
    }}
    .modal-content {{
      position: relative;
      max-width: min(92vw, 1400px);
      max-height: 90vh;
      width: 100%;
      background: #ffffff;
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
    }}
    .modal-content img {{
      width: 100%;
      max-height: 78vh;
      object-fit: contain;
      border-radius: 12px;
    }}
    .modal-close {{
      position: absolute;
      top: 8px;
      right: 10px;
      border: 0;
      background: transparent;
      font-size: 1.8rem;
      cursor: pointer;
      color: #374151;
    }}
    .modal-title {{
      margin: 0 36px 12px 0;
      font-size: 1rem;
      font-weight: 700;
      color: #374151;
    }}
    @media (max-width: 980px) {{
      .two-col {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>FINMEM Comparison Dashboard</h1>
      <p class="subtle">This dashboard consolidates the currently available outputs and adds explicit placeholders where runs are not finished yet. Click any chart to magnify it. Use the collapsible panels to keep the page compact while reviewing tables and notes.</p>
      <div class="links">
        <a href="../{_rel(comparison1_csv)}">Comparison 1 CSV</a>
        <a href="../{_rel(comparison2_csv)}">RQ2 CSV</a>
        <a href="../{_rel(comparison3_csv)}">TSLA Trading Agents CSV</a>
        <a href="../{_rel(comparison3_risk_csv)}">TSLA Risk Profiles CSV</a>
        <a href="../{_rel(comparison3_risk_gpt35_csv)}">TSLA Risk Profiles GPT-3.5 CSV</a>
        <a href="../{_rel(comparison4_csv)}">TSLA Top-K CSV</a>
      </div>
    </section>

    <section class="card section">
      <h2>Data Section Placeholder</h2>
      <p>The data section is not finalized yet, so these remain placeholders in the dashboard.</p>
      <div class="placeholder-grid">
        <div class="placeholder"><strong>Price Data</strong><br />Placeholder</div>
        <div class="placeholder"><strong>News Data</strong><br />Placeholder</div>
        <div class="placeholder"><strong>10-K / 10-Q</strong><br />Placeholder</div>
        <div class="placeholder"><strong>Sentiment</strong><br />Placeholder</div>
        <div class="placeholder"><strong>Merged Env Data</strong><br />Placeholder</div>
      </div>
    </section>

    <section class="section">
      <span class="section-label">RQ1</span>
      <h2>Does FINMEM outperform the comparison agents?</h2>
      <p>Per paper structure, this section is ticker-by-ticker. Each card contains one cumulative-return chart and one metrics table for Buy &amp; Hold, the current FinMem run, the new self-adaptive paper-aligned FinMem run, FinGPT, Generative Agents, A2C, PPO, and DQN.</p>
      <div class="grid">
        {''.join(ticker_cards)}
      </div>
    </section>

    <section class="section two-col">
      <section class="card">
        <span class="section-label">RQ2</span>
        <h2>Does FINMEM remain effective with limited training data?</h2>
        <details class="details-block" open>
          <summary>Show Limited-Training Chart</summary>
          <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison2_chart)}' data-title='TSLA limited training chart'>
            <img src="../{_rel(comparison2_chart)}" alt="TSLA limited training chart" />
          </button>
        </details>
      </section>

      <section class="card">
        <span class="section-label">RQ2 Table</span>
        <h2>TSLA Full vs Limited Training</h2>
        <details class="details-block" open>
          <summary>Show Limited-Training Table</summary>
          <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison2_svg)}' data-title='TSLA limited training table'>
            <img src="../{_rel(comparison2_svg)}" alt="TSLA limited training table" />
          </button>
        </details>
      </section>
    </section>

    <section class="section two-col">
      <section class="card">
        <span class="section-label">RQ3</span>
        <h2>Which LLM backbone works best for FINMEM?</h2>
        <details class="details-block" open>
          <summary>Show TSLA Trading Agents Chart</summary>
          <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison3_chart)}' data-title='TSLA trading agents chart'>
            <img src="../{_rel(comparison3_chart)}" alt="TSLA trading agents chart" />
          </button>
        </details>
      </section>

      <section class="card">
        <span class="section-label">RQ3 Table</span>
        <h2>TSLA Trading Agents Metrics</h2>
        <p>Metrics: Cumulative Return (%), Sharpe Ratio, Daily Volatility (%), Annualized Volatility (%), Max Drawdown (%).</p>
        <p>Paper note: in the Trading Agents Comparison, FINMEM employs GPT-4-Turbo as its backbone algorithm with temperature 0.7. The table below shows current local outputs and placeholders where those runs are still missing.</p>
        <details class="details-block" open>
          <summary>Show TSLA Trading Agents Table</summary>
          <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison3_svg)}' data-title='TSLA trading agents metrics table'>
            <img src="../{_rel(comparison3_svg)}" alt="TSLA trading agents metrics table" />
          </button>
        </details>
      </section>

    </section>

    <section class="section two-col">
      <section class="card">
        <span class="section-label">RQ4A</span>
        <h2>Risk Profiles - GPT-4-Turbo</h2>
        <p>{html.escape(comparison3_risk_conclusion)}</p>
        <details class="details-block" open>
          <summary>Show GPT-4-Turbo Risk Chart</summary>
          <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison3_risk_chart)}' data-title='TSLA GPT-4-Turbo risk profile chart'>
            <img src="../{_rel(comparison3_risk_chart)}" alt="TSLA GPT-4-Turbo risk profile chart" />
          </button>
        </details>
        <details class="details-block">
          <summary>Show GPT-4-Turbo Risk Table</summary>
          <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison3_risk_svg)}' data-title='TSLA GPT-4-Turbo risk profile table'>
            <img src="../{_rel(comparison3_risk_svg)}" alt="TSLA GPT-4-Turbo risk profile table" />
          </button>
        </details>
      </section>

      <section class="card">
        <span class="section-label">RQ4B</span>
        <h2>Risk Profiles - GPT-3.5-Turbo</h2>
        <p>{html.escape(comparison3_risk_gpt35_conclusion)}</p>
        <details class="details-block" open>
          <summary>Show GPT-3.5 Risk Chart</summary>
          <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison3_risk_gpt35_chart)}' data-title='TSLA GPT-3.5 risk profile chart'>
            <img src="../{_rel(comparison3_risk_gpt35_chart)}" alt="TSLA GPT-3.5 risk profile chart" />
          </button>
        </details>
        <details class="details-block">
          <summary>Show GPT-3.5 Risk Table</summary>
          <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison3_risk_gpt35_svg)}' data-title='TSLA GPT-3.5 risk profile table'>
            <img src="../{_rel(comparison3_risk_gpt35_svg)}" alt="TSLA GPT-3.5 risk profile table" />
          </button>
        </details>
      </section>
    </section>

    <section class="card section">
      <span class="section-label">RQ5</span>
      <h2>Top 1 / Top 3 / Top 5 / Top 10</h2>
      <details class="details-block" open>
        <summary>Show Top-K Chart</summary>
        <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison4_chart)}' data-title='TSLA top-k chart'>
          <img src="../{_rel(comparison4_chart)}" alt="TSLA top-k chart" />
        </button>
      </details>
      <details class="details-block">
        <summary>Show Top-K Table</summary>
        <button class='chart-button' type='button' data-fullsrc='../{_rel(comparison4_svg)}' data-title='TSLA top-k table'>
          <img src="../{_rel(comparison4_svg)}" alt="TSLA top-k table" />
        </button>
      </details>
    </section>
  </div>

  <div id="chart-modal" class="modal" aria-hidden="true">
    <div class="modal-content">
      <button id="chart-modal-close" class="modal-close" type="button" aria-label="Close">×</button>
      <div id="chart-modal-title" class="modal-title"></div>
      <img id="chart-modal-image" src="" alt="" />
    </div>
  </div>

  <script>
    (function() {{
      const modal = document.getElementById('chart-modal');
      const modalImg = document.getElementById('chart-modal-image');
      const modalTitle = document.getElementById('chart-modal-title');
      const closeBtn = document.getElementById('chart-modal-close');

      function closeModal() {{
        modal.classList.remove('open');
        modal.setAttribute('aria-hidden', 'true');
        modalImg.setAttribute('src', '');
        modalImg.setAttribute('alt', '');
        modalTitle.textContent = '';
      }}

      document.querySelectorAll('.chart-button').forEach((button) => {{
        button.addEventListener('click', () => {{
          const src = button.getAttribute('data-fullsrc');
          const title = button.getAttribute('data-title') || 'Chart preview';
          modalImg.setAttribute('src', src);
          modalImg.setAttribute('alt', title);
          modalTitle.textContent = title;
          modal.classList.add('open');
          modal.setAttribute('aria-hidden', 'false');
        }});
      }});

      closeBtn.addEventListener('click', closeModal);
      modal.addEventListener('click', (event) => {{
        if (event.target === modal) {{
          closeModal();
        }}
      }});
      document.addEventListener('keydown', (event) => {{
        if (event.key === 'Escape') {{
          closeModal();
        }}
      }});
    }})();
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate requested FINMEM comparison dashboard assets.")
    parser.add_argument(
        "--output",
        default="dashboard/results_dashboard.html",
        help="Output HTML path.",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)

    tsla_prices = _load_prices("TSLA", TEST_START, TEST_END)
    comparison1 = _build_comparison1_outputs()
    comparison1_csv = OUTPUT_ROOT / "comparison_1_all_tickers_metrics.csv"
    comparison2_csv, comparison2_svg, comparison2_chart = _build_tsla_limited_training(tsla_prices)
    _, comparison3_csv, comparison3_svg, comparison3_chart = _build_tsla_trading_agents(tsla_prices)
    comparison3_risk_csv, comparison3_risk_svg, comparison3_risk_chart, comparison3_risk_conclusion = _build_tsla_risk_profiles(
        tsla_prices,
        TSLA_RISK_PATHS,
        "gpt4",
        "GPT-4-Turbo (temp=0.7, top_k=5)",
        "GPT-4-Turbo risk-profile runs are still incomplete. Placeholder zero-lines remain for missing outputs.",
    )
    comparison3_risk_gpt35_csv, comparison3_risk_gpt35_svg, comparison3_risk_gpt35_chart, comparison3_risk_gpt35_conclusion = _build_tsla_risk_profiles(
        tsla_prices,
        TSLA_GPT35_RISK_PATHS,
        "gpt35",
        "GPT-3.5-Turbo (temp=0.7, top_k=5, look_back_window_size=3)",
        "GPT-3.5 risk-profile runs are still incomplete. Placeholder zero-lines remain for missing outputs.",
    )
    comparison4_csv, comparison4_svg, comparison4_chart = _build_tsla_topk(tsla_prices)

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_html(
            comparison1,
            comparison1_csv,
            comparison2_csv,
            comparison2_svg,
            comparison2_chart,
            comparison3_csv,
            comparison3_svg,
            comparison3_chart,
            comparison3_risk_csv,
            comparison3_risk_svg,
            comparison3_risk_chart,
            comparison3_risk_conclusion,
            comparison3_risk_gpt35_csv,
            comparison3_risk_gpt35_svg,
            comparison3_risk_gpt35_chart,
            comparison3_risk_gpt35_conclusion,
            comparison4_csv,
            comparison4_svg,
            comparison4_chart,
        ),
        encoding="utf-8",
    )
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    main()

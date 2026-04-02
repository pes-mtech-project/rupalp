import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_baselines.common import (
    ACTION_MAP,
    build_llm_client,
    compute_metrics,
    load_resume_state,
    load_subset_frame,
    parse_decision,
    recent_return_summary,
    split_frame,
    truncate_news,
)


def build_prompt(ticker: str, context_df: pd.DataFrame) -> str:
    current = context_df.iloc[-1]
    recent_prices = recent_return_summary(context_df[["date", "close"]].reset_index(drop=True))
    news_block = "\n".join(
        f"- {item}" for item in truncate_news(current["news"], max_items=8, max_chars=300)
    )
    filing_bits = []
    if current["filing_q"]:
        filing_bits.append(f"10-Q: {str(current['filing_q'])[:600]}")
    if current["filing_k"]:
        filing_bits.append(f"10-K: {str(current['filing_k'])[:600]}")
    filing_block = "\n".join(filing_bits) if filing_bits else "None"

    return f"""
You are a Generative Agents style single-stock trader.
You do not have layered memory like FinMem. Use only the recent market diary below.

Ticker: {ticker}
Current date: {current['date'].date().isoformat()}

Recent price diary:
{recent_prices}

Today's news:
{news_block if news_block else 'None'}

Today's filings:
{filing_block}

Decide the NEXT trading day action for {ticker}.
Return JSON only:
{{"decision":"buy|hold|sell","summary":"one short reason"}}
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Generative Agents style baseline on single-stock test data.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--subset-pkl", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--model", default="gpt-3.5-turbo-0125")
    parser.add_argument("--end-point", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--lookback-window-size", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions_csv = output_dir / f"{args.ticker}_ga_actions.csv"
    metrics_json = output_dir / f"{args.ticker}_ga_metrics.json"

    llm = build_llm_client(
        model=args.model,
        end_point=args.end_point,
        system_message="You are a disciplined trading assistant.",
        temperature=0.0,
    )

    df = split_frame(load_subset_frame(args.subset_pkl, args.ticker), args.test_start, args.test_end)
    saved_rows, start_idx, cumulative = load_resume_state(actions_csv)
    rows = list(saved_rows)

    for i in range(start_idx, len(df) - 1):
        context_df = df.iloc[max(0, i - args.lookback_window_size + 1) : i + 1].reset_index(drop=True)
        raw_response = llm(build_prompt(args.ticker, context_df))
        decision, summary = parse_decision(raw_response)
        action = ACTION_MAP[decision]
        cur_price = float(df.iloc[i]["close"])
        next_price = float(df.iloc[i + 1]["close"])
        reward = float(action * np.log(next_price / cur_price))
        cumulative += reward
        rows.append(
            {
                "date": df.iloc[i]["date"].date().isoformat(),
                "action": action,
                "decision": decision,
                "summary": summary,
                "reward": reward,
                "cumulative_return": cumulative,
                "close": cur_price,
                "next_close": next_price,
                "raw_response": raw_response,
            }
        )
        pd.DataFrame(rows).to_csv(actions_csv, index=False)

    result_df = pd.DataFrame(rows)
    metrics = {"ticker": args.ticker, "baseline": "GA", "model": args.model, **compute_metrics(result_df)}
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

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


def _sentiment_hint(news_items) -> str:
    joined = " ".join(str(x) for x in news_items)
    pos_hits = joined.lower().count("positive score")
    neg_hits = joined.lower().count("negative score")
    return f"sentiment_mentions_positive={pos_hits}, sentiment_mentions_negative={neg_hits}"


def build_prompt(ticker: str, context_df: pd.DataFrame) -> str:
    current = context_df.iloc[-1]
    recent_prices = recent_return_summary(context_df[["date", "close"]].reset_index(drop=True))
    news_items = truncate_news(current["news"], max_items=10, max_chars=320)
    news_block = "\n".join(f"- {item}" for item in news_items)
    filing_q = str(current["filing_q"])[:900] if current["filing_q"] else "None"
    filing_k = str(current["filing_k"])[:900] if current["filing_k"] else "None"
    sentiment_hint = _sentiment_hint(current["news"])

    return f"""
You are a FinGPT-style financial analyst focused on single-stock trading.
Use the structured financial evidence to forecast the NEXT trading day move for {ticker}.
Be conservative when evidence is mixed.

Ticker: {ticker}
Current date: {current['date'].date().isoformat()}

Recent prices:
{recent_prices}

News headlines:
{news_block if news_block else 'None'}

Sentiment hints:
{sentiment_hint}

10-Q excerpt:
{filing_q}

10-K excerpt:
{filing_k}

Return JSON only:
{{"decision":"buy|hold|sell","summary":"one short financial reason"}}
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a FinGPT-style baseline on single-stock test data.")
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
    actions_csv = output_dir / f"{args.ticker}_fingpt_actions.csv"
    metrics_json = output_dir / f"{args.ticker}_fingpt_metrics.json"

    llm = build_llm_client(
        model=args.model,
        end_point=args.end_point,
        system_message="You are a careful financial analysis assistant.",
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
    metrics = {"ticker": args.ticker, "baseline": "FinGPT", "model": args.model, **compute_metrics(result_df)}
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

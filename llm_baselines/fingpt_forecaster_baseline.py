import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_baselines.common import (
    ACTION_MAP,
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
    recent_prices = recent_return_summary(
        context_df[["date", "close"]].reset_index(drop=True)
    )
    news_items = truncate_news(current["news"], max_items=10, max_chars=320)
    news_block = "\n".join(f"- {item}" for item in news_items)
    filing_q = str(current["filing_q"])[:900] if current["filing_q"] else "None"
    filing_k = str(current["filing_k"])[:900] if current["filing_k"] else "None"

    return f"""
[INST]
You are FinGPT Forecaster, a financial large language model for stock movement analysis.
Given the recent trading context for {ticker}, decide the NEXT trading day action.
Be conservative when evidence is weak or contradictory.

Ticker: {ticker}
Current date: {current['date'].date().isoformat()}

Recent prices:
{recent_prices}

Recent news:
{news_block if news_block else 'None'}

10-Q excerpt:
{filing_q}

10-K excerpt:
{filing_k}

Return JSON only:
{{"decision":"buy|hold|sell","summary":"one short financial reason"}}
[/INST]
""".strip()


def _resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    return (
        explicit_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def _resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float16


def _materialize_repo(
    repo_id: str,
    hf_token: Optional[str],
    local_root: Path,
) -> str:
    local_root.mkdir(parents=True, exist_ok=True)
    safe_name = repo_id.replace("/", "__")
    local_dir = local_root / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        token=hf_token,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        max_workers=1,
        resume_download=True,
    )


def load_fingpt_model(
    base_model_name: str,
    adapter_model_name: str,
    hf_token: Optional[str],
):
    hf_home = Path(
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or "/tmp/hf_cache"
    )
    model_root = hf_home / "materialized_models"
    base_model_path = _materialize_repo(base_model_name, hf_token, model_root)
    adapter_model_path = _materialize_repo(adapter_model_name, hf_token, model_root)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        token=hf_token,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_auto_device_map = torch.cuda.is_available()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto" if use_auto_device_map else None,
        dtype=_resolve_dtype(),
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    if not use_auto_device_map:
        base_model = base_model.to("cpu")
    model = PeftModel.from_pretrained(base_model, adapter_model_path, token=hf_token)
    model = model.eval()
    return tokenizer, model


def infer_text(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 2048), 2048),
    )
    model_device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(model_device)
    attention_mask = inputs["attention_mask"].to(model_device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][input_ids.shape[1] :]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official HF FinGPT forecaster LoRA baseline on single-stock test data."
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--subset-pkl", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument(
        "--adapter-model",
        default="FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--lookback-window-size", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions_csv = output_dir / f"{args.ticker}_fingpt_actions.csv"
    metrics_json = output_dir / f"{args.ticker}_fingpt_metrics.json"

    hf_token = _resolve_hf_token(args.hf_token)
    if args.base_model.startswith("meta-llama/") and not hf_token:
        raise ValueError(
            "Missing Hugging Face token. Set HF_TOKEN (or pass --hf-token) before running FinGPT."
        )

    tokenizer, model = load_fingpt_model(args.base_model, args.adapter_model, hf_token)

    df = split_frame(
        load_subset_frame(args.subset_pkl, args.ticker), args.test_start, args.test_end
    )
    saved_rows, start_idx, cumulative = load_resume_state(actions_csv)
    rows = list(saved_rows)

    for i in range(start_idx, len(df) - 1):
        context_df = df.iloc[
            max(0, i - args.lookback_window_size + 1) : i + 1
        ].reset_index(drop=True)
        raw_response = infer_text(
            tokenizer,
            model,
            build_prompt(args.ticker, context_df),
            args.max_new_tokens,
        )
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
    metrics = {
        "ticker": args.ticker,
        "baseline": "FinGPT",
        "base_model": args.base_model,
        "adapter_model": args.adapter_model,
        **compute_metrics(result_df),
    }
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

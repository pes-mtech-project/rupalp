#!/usr/bin/env bash
set -euo pipefail

TICKERS="${TICKERS:-TSLA AMZN NFLX MSFT COIN}"
TEST_START="${TEST_START:-2026-01-02}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
ADAPTER_MODEL="${ADAPTER_MODEL:-FinGPT/fingpt-forecaster_dow30_llama2-7b_lora}"
LOOKBACK_WINDOW_SIZE="${LOOKBACK_WINDOW_SIZE:-7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}" ]]; then
  echo "Missing Hugging Face token. Export HF_TOKEN before running this script."
  exit 1
fi

clip_end_date() {
  python3 - "$1" "${TEST_END:-$(date +%F)}" <<'PY'
import pickle
import sys
from datetime import datetime

path, requested = sys.argv[1], sys.argv[2]
with open(path, "rb") as f:
    env = pickle.load(f)
max_date = max(env.keys())
requested_dt = datetime.strptime(requested, "%Y-%m-%d").date()
print(min(max_date, requested_dt).isoformat())
PY
}

for ticker in $TICKERS; do
  subset="data/06_input/subset_symbols_${ticker}.pkl"
  if [[ ! -f "$subset" ]]; then
    echo "Skipping $ticker: missing $subset"
    continue
  fi
  end_date="$(clip_end_date "$subset")"

  echo "== $ticker / FinGPT Forecaster =="
  mkdir -p "data/llm_baselines/fingpt_hf/${ticker}"
  python3 -m llm_baselines.fingpt_forecaster_baseline \
    --ticker "$ticker" \
    --subset-pkl "$subset" \
    --test-start "$TEST_START" \
    --test-end "$end_date" \
    --base-model "$BASE_MODEL" \
    --adapter-model "$ADAPTER_MODEL" \
    --lookback-window-size "$LOOKBACK_WINDOW_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-dir "data/llm_baselines/fingpt_hf/${ticker}"
done

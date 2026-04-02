#!/usr/bin/env bash
set -euo pipefail

TICKERS="${TICKERS:-TSLA AMZN NFLX MSFT COIN}"
TEST_START="${TEST_START:-2026-01-02}"
MODEL="${MODEL:-gpt-3.5-turbo-0125}"
END_POINT="${END_POINT:-https://api.openai.com/v1/chat/completions}"
LOOKBACK_WINDOW_SIZE="${LOOKBACK_WINDOW_SIZE:-7}"

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

  echo "== $ticker / FinGPT-style =="
  mkdir -p "data/llm_baselines/fingpt/${ticker}"
  python3 -m llm_baselines.fingpt_baseline \
    --ticker "$ticker" \
    --subset-pkl "$subset" \
    --test-start "$TEST_START" \
    --test-end "$end_date" \
    --model "$MODEL" \
    --end-point "$END_POINT" \
    --lookback-window-size "$LOOKBACK_WINDOW_SIZE" \
    --output-dir "data/llm_baselines/fingpt/${ticker}"
done

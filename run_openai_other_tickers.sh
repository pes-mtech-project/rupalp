#!/usr/bin/env bash
set -euo pipefail

START_TRAIN="${START_TRAIN:-2025-01-02}"
END_TRAIN="${END_TRAIN:-2025-12-31}"
START_TEST="${START_TEST:-2026-01-02}"
END_TEST="${END_TEST:-$(date +%F)}"

for ticker in AMZN NFLX MSFT COIN; do
  lc="$(printf '%s' "$ticker" | tr '[:upper:]' '[:lower:]')"
  effective_end_test="$(
    python3 - "$ticker" "$END_TEST" <<'PY'
import pickle
import sys
from pathlib import Path

ticker = sys.argv[1]
requested_end = sys.argv[2]
pkl_path = Path(f"data/06_input/subset_symbols_{ticker}.pkl")
with open(pkl_path, "rb") as f:
    env = pickle.load(f)
max_date = max(env.keys()).isoformat()
print(min(requested_end, max_date))
PY
  )"

  mkdir -p \
    "data/06_train_checkpoint/${lc}_2025" \
    "data/05_train_model_output/${lc}_2025" \
    "data/06_test_checkpoint/${lc}_2026" \
    "data/05_test_model_output/${lc}_2026"

  echo "== ${ticker}: train =="
  python3 run.py sim \
    --market-data-path "data/06_input/subset_symbols_${ticker}.pkl" \
    --start-time "${START_TRAIN}" \
    --end-time "${END_TRAIN}" \
    --run-model train \
    --config-path "config/${lc}_gpt_config.toml" \
    --checkpoint-path "data/06_train_checkpoint/${lc}_2025" \
    --result-path "data/05_train_model_output/${lc}_2025"

  echo "== ${ticker}: test =="
  python3 run.py sim \
    --market-data-path "data/06_input/subset_symbols_${ticker}.pkl" \
    --start-time "${START_TEST}" \
    --end-time "${effective_end_test}" \
    --run-model test \
    --config-path "config/${lc}_gpt_config.toml" \
    --trained-agent-path "data/05_train_model_output/${lc}_2025" \
    --checkpoint-path "data/06_test_checkpoint/${lc}_2026" \
    --result-path "data/05_test_model_output/${lc}_2026"
done

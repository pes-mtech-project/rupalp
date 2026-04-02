#!/usr/bin/env bash
set -euo pipefail

TRAIN_START="${TRAIN_START:-2025-01-02}"
TRAIN_END="${TRAIN_END:-2025-12-31}"
TEST_START="${TEST_START:-2026-01-02}"
TEST_END="${TEST_END:-2026-03-06}"
TIMESTEPS="${TIMESTEPS:-50000}"

for ticker in TSLA AMZN NFLX MSFT COIN; do
  sentiment_csv=""
  if [ -f "data/06_input/subset_symbols_${ticker}_sentiment.csv" ]; then
    sentiment_csv="--sentiment-csv data/06_input/subset_symbols_${ticker}_sentiment.csv"
  fi

  for algo in ppo dqn a2c; do
    out_dir="data/drl_results/$(printf '%s' "$ticker" | tr '[:upper:]' '[:lower:]')_${algo}"
    mkdir -p "$out_dir"
    echo "== ${ticker} / ${algo} =="
    python3 -m drl_baselines.train_eval \
      --ticker "${ticker}" \
      --subset-pkl "data/06_input/subset_symbols_${ticker}.pkl" \
      --algo "${algo}" \
      --train-start "${TRAIN_START}" \
      --train-end "${TRAIN_END}" \
      --test-start "${TEST_START}" \
      --test-end "${TEST_END}" \
      --timesteps "${TIMESTEPS}" \
      ${sentiment_csv} \
      --output-dir "${out_dir}"
  done
done

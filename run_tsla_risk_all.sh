#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TRAIN_START="${TRAIN_START:-2025-01-02}"
TRAIN_END="${TRAIN_END:-2025-12-31}"
TEST_START="${TEST_START:-2026-01-02}"
TEST_END="${TEST_END:-2026-03-06}"
VARIANTS="${VARIANTS:-risk_averse risk_seeking self_adaptive}"
MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/06_input/subset_symbols_TSLA.pkl}"

for variant in $VARIANTS; do
  config_path="config/tsla_gpt_${variant}_config.toml"
  train_checkpoint="data/06_train_checkpoint/tsla_${variant}"
  train_result="data/05_train_model_output/tsla_${variant}"
  test_checkpoint="data/06_test_checkpoint/tsla_${variant}"
  test_result="data/05_test_model_output/tsla_${variant}"

  if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path"
    echo "Run: python3 prepare_tsla_risk_variants.py"
    exit 1
  fi

  mkdir -p "$train_checkpoint" "$train_result" "$test_checkpoint" "$test_result"

  echo "== TSLA ${variant}: train =="
  python3 run.py sim \
    --market-data-path "$MARKET_DATA_PATH" \
    --start-time "$TRAIN_START" \
    --end-time "$TRAIN_END" \
    --run-model train \
    --config-path "$config_path" \
    --checkpoint-path "$train_checkpoint" \
    --result-path "$train_result"

  echo "== TSLA ${variant}: test =="
  python3 run.py sim \
    --market-data-path "$MARKET_DATA_PATH" \
    --start-time "$TEST_START" \
    --end-time "$TEST_END" \
    --run-model test \
    --config-path "$config_path" \
    --trained-agent-path "$train_result" \
    --checkpoint-path "$test_checkpoint" \
    --result-path "$test_result"
done

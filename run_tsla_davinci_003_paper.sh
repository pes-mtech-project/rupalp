#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TRAIN_START="${TRAIN_START:-2025-01-02}"
TRAIN_END="${TRAIN_END:-2025-12-31}"
TEST_START="${TEST_START:-2026-01-02}"
TEST_END="${TEST_END:-2026-03-06}"
MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/06_input/subset_symbols_TSLA.pkl}"
CONFIG_PATH="${CONFIG_PATH:-config/tsla_davinci_003_paper_config.toml}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-data/06_train_checkpoint/tsla_davinci_003_paper}"
TRAIN_RESULT="${TRAIN_RESULT:-data/05_train_model_output/tsla_davinci_003_paper}"
TEST_CHECKPOINT="${TEST_CHECKPOINT:-data/06_test_checkpoint/tsla_davinci_003_paper}"
TEST_RESULT="${TEST_RESULT:-data/05_test_model_output/tsla_davinci_003_paper}"

mkdir -p "$TRAIN_CHECKPOINT" "$TRAIN_RESULT" "$TEST_CHECKPOINT" "$TEST_RESULT"

echo "== TSLA davinci-003 paper-aligned: train =="
python3 run.py sim \
  --market-data-path "$MARKET_DATA_PATH" \
  --start-time "$TRAIN_START" \
  --end-time "$TRAIN_END" \
  --run-model train \
  --config-path "$CONFIG_PATH" \
  --checkpoint-path "$TRAIN_CHECKPOINT" \
  --result-path "$TRAIN_RESULT"

echo "== TSLA davinci-003 paper-aligned: test =="
python3 run.py sim \
  --market-data-path "$MARKET_DATA_PATH" \
  --start-time "$TEST_START" \
  --end-time "$TEST_END" \
  --run-model test \
  --config-path "$CONFIG_PATH" \
  --trained-agent-path "$TRAIN_RESULT" \
  --checkpoint-path "$TEST_CHECKPOINT" \
  --result-path "$TEST_RESULT"

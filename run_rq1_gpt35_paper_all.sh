#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TRAIN_START="${TRAIN_START:-2025-01-02}"
TRAIN_END="${TRAIN_END:-2025-12-31}"
TEST_START="${TEST_START:-2026-01-02}"
TEST_END="${TEST_END:-2026-03-06}"
TICKERS="${TICKERS:-TSLA AMZN NFLX MSFT COIN}"

for ticker in $TICKERS; do
  lower="$(echo "$ticker" | tr '[:upper:]' '[:lower:]')"
  config_path="config/${lower}_gpt_rq1_paper_config.toml"
  market_data="data/06_input/subset_symbols_${ticker}.pkl"
  train_checkpoint="data/06_train_checkpoint/${lower}_rq1_paper_gpt35"
  train_result="data/05_train_model_output/${lower}_rq1_paper_gpt35"
  test_checkpoint="data/06_test_checkpoint/${lower}_rq1_paper_gpt35"
  test_result="data/05_test_model_output/${lower}_rq1_paper_gpt35"

  if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path"
    echo "Run: python3 prepare_rq1_gpt35_paper_configs.py"
    exit 1
  fi

  mkdir -p "$train_checkpoint" "$train_result" "$test_checkpoint" "$test_result"

  echo "== ${ticker} RQ1 paper-aligned GPT-3.5: train =="
  python3 run.py sim \
    --market-data-path "$market_data" \
    --start-time "$TRAIN_START" \
    --end-time "$TRAIN_END" \
    --run-model train \
    --config-path "$config_path" \
    --checkpoint-path "$train_checkpoint" \
    --result-path "$train_result"

  echo "== ${ticker} RQ1 paper-aligned GPT-3.5: test =="
  python3 run.py sim \
    --market-data-path "$market_data" \
    --start-time "$TEST_START" \
    --end-time "$TEST_END" \
    --run-model test \
    --config-path "$config_path" \
    --trained-agent-path "$train_result" \
    --checkpoint-path "$test_checkpoint" \
    --result-path "$test_result"
done

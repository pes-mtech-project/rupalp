#!/usr/bin/env bash
set -euo pipefail

MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/06_input/subset_symbols_TSLA.pkl}"
TRAIN_START="${TRAIN_START:-2025-01-02}"
TRAIN_END="${TRAIN_END:-2025-12-31}"
TEST_START="${TEST_START:-2026-01-02}"
RAW_TEST_END="${TEST_END:-$(date +%F)}"
MODELS="${MODELS:-gpt35 gpt4 gpt4_turbo davinci_003}"
RUN_LLAMA2="${RUN_LLAMA2:-0}"

TEST_END="$(python3 - "$MARKET_DATA_PATH" "$RAW_TEST_END" <<'PY'
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
)"

if [[ "$RUN_LLAMA2" == "1" ]]; then
  MODELS="$MODELS llama2_70b_chat"
fi

run_model() {
  local key="$1"
  local label config train_ckpt train_out test_ckpt test_out

  case "$key" in
    gpt35)
      label="GPT-3.5-Turbo"
      config="config/tsla_gpt35_paper_config.toml"
      train_ckpt="data/06_train_checkpoint/tsla_gpt35"
      train_out="data/05_train_model_output/tsla_gpt35"
      test_ckpt="data/06_test_checkpoint/tsla_gpt35"
      test_out="data/05_test_model_output/tsla_gpt35"
      ;;
    gpt4)
      label="GPT-4"
      config="config/tsla_gpt4_config.toml"
      train_ckpt="data/06_train_checkpoint/tsla_gpt4"
      train_out="data/05_train_model_output/tsla_gpt4"
      test_ckpt="data/06_test_checkpoint/tsla_gpt4"
      test_out="data/05_test_model_output/tsla_gpt4"
      ;;
    gpt4_turbo)
      label="GPT-4-Turbo"
      config="config/tsla_gpt4_turbo_config.toml"
      train_ckpt="data/06_train_checkpoint/tsla_gpt4_turbo"
      train_out="data/05_train_model_output/tsla_gpt4_turbo"
      test_ckpt="data/06_test_checkpoint/tsla_gpt4_turbo"
      test_out="data/05_test_model_output/tsla_gpt4_turbo"
      ;;
    davinci_003)
      label="davinci-003"
      config="config/tsla_davinci_003_config.toml"
      train_ckpt="data/06_train_checkpoint/tsla_davinci_003"
      train_out="data/05_train_model_output/tsla_davinci_003"
      test_ckpt="data/06_test_checkpoint/tsla_davinci_003"
      test_out="data/05_test_model_output/tsla_davinci_003"
      ;;
    llama2_70b_chat)
      label="Llama2-70b-chat"
      config="config/tsla_llama2_70b_chat_config.toml"
      train_ckpt="data/06_train_checkpoint/tsla_llama2_70b_chat"
      train_out="data/05_train_model_output/tsla_llama2_70b_chat"
      test_ckpt="data/06_test_checkpoint/tsla_llama2_70b_chat"
      test_out="data/05_test_model_output/tsla_llama2_70b_chat"
      ;;
    *)
      echo "Unknown model key: $key" >&2
      exit 1
      ;;
  esac

  echo "== $label: train =="
  mkdir -p "$train_ckpt" "$train_out" "$test_ckpt" "$test_out"
  python3 run.py sim \
    --market-data-path "$MARKET_DATA_PATH" \
    --start-time "$TRAIN_START" \
    --end-time "$TRAIN_END" \
    --run-model train \
    --config-path "$config" \
    --checkpoint-path "$train_ckpt" \
    --result-path "$train_out"

  echo "== $label: test =="
  python3 run.py sim \
    --market-data-path "$MARKET_DATA_PATH" \
    --start-time "$TEST_START" \
    --end-time "$TEST_END" \
    --run-model test \
    --config-path "$config" \
    --trained-agent-path "$train_out" \
    --checkpoint-path "$test_ckpt" \
    --result-path "$test_out"
}

for key in $MODELS; do
  run_model "$key"
done

echo "Done. Test window used: $TEST_START to $TEST_END"

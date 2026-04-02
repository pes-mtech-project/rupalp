#!/usr/bin/env bash
set -euo pipefail

MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/06_input/subset_symbols_TSLA.pkl}"
TEST_START="${TEST_START:-2026-01-02}"
RAW_TEST_END="${TEST_END:-$(date +%F)}"

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

MODEL_ARGS=()

add_model() {
  local name="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    MODEL_ARGS+=("--model" "${name}=${path}")
  else
    echo "Skipping missing model output: $name ($path)"
  fi
}

add_model "GPT-3.5-Turbo" "data/05_test_model_output/tsla_gpt35/agent_1/state_dict.pkl"
add_model "GPT-4" "data/05_test_model_output/tsla_gpt4/agent_1/state_dict.pkl"
add_model "GPT-4-Turbo" "data/05_test_model_output/tsla_gpt4_turbo/agent_1/state_dict.pkl"
add_model "davinci-003" "data/05_test_model_output/tsla_davinci_003/agent_1/state_dict.pkl"
add_model "Llama2-70b-chat" "data/05_test_model_output/tsla_llama2_70b_chat/agent_1/state_dict.pkl"

if [[ "${#MODEL_ARGS[@]}" -eq 0 ]]; then
  echo "No model outputs found to compare." >&2
  exit 1
fi

python3 data-pipeline/07-metrics_new.py \
  --env-pkl "$MARKET_DATA_PATH" \
  --ticker TSLA \
  --start-date "$TEST_START" \
  --end-date "$TEST_END" \
  "${MODEL_ARGS[@]}" \
  --output data/07_model_output/tsla_paper_models_metrics.csv

python3 data-pipeline/06-Visualize-results_new.py \
  --env-pkl "$MARKET_DATA_PATH" \
  --ticker TSLA \
  --start-date "$TEST_START" \
  --end-date "$TEST_END" \
  "${MODEL_ARGS[@]}" \
  --output data/07_model_output/tsla_paper_models_cumulative_returns.png

python3 data-pipeline/06b-export-cumulative-series_new.py \
  --env-pkl "$MARKET_DATA_PATH" \
  --ticker TSLA \
  --start-date "$TEST_START" \
  --end-date "$TEST_END" \
  "${MODEL_ARGS[@]}" \
  --output data/07_model_output/tsla_paper_models_cumulative_series.csv

python3 data-pipeline/08-Wilcoxon-Test_new.py \
  --env-pkl "$MARKET_DATA_PATH" \
  --ticker TSLA \
  --start-date "$TEST_START" \
  --end-date "$TEST_END" \
  "${MODEL_ARGS[@]}" \
  --output data/07_model_output/tsla_paper_models_wilcoxon.csv

echo "Saved:"
echo "  data/07_model_output/tsla_paper_models_metrics.csv"
echo "  data/07_model_output/tsla_paper_models_cumulative_returns.png"
echo "  data/07_model_output/tsla_paper_models_cumulative_series.csv"
echo "  data/07_model_output/tsla_paper_models_wilcoxon.csv"

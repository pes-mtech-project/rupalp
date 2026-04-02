import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = REPO_ROOT / "config/tsla_gpt_config.toml"

TOPK_VARIANTS = {
    1: "config/tsla_gpt_top1_config.toml",
    3: "config/tsla_gpt_top3_config.toml",
    5: "config/tsla_gpt_top5_config.toml",
    10: "config/tsla_gpt_top10_config.toml",
}


def _replace_top_k(text: str, top_k: int) -> str:
    updated, count = re.subn(r"(?m)^top_k\s*=\s*\d+\s*$", f"top_k = {top_k}", text, count=1)
    if count != 1:
        raise ValueError("Could not find top_k line in base config.")
    return (
        f"# Auto-generated from {BASE_CONFIG.name}\n"
        f"# Variant: TSLA top_k={top_k}\n\n"
        + updated
    )


def _print_commands(config_path: str, suffix: str) -> None:
    print(f"\n[{suffix}]")
    print(
        "python3 run.py sim "
        "--market-data-path data/06_input/subset_symbols_TSLA.pkl "
        "--start-time 2025-01-02 "
        "--end-time 2025-12-31 "
        "--run-model train "
        f"--config-path {config_path} "
        f"--checkpoint-path data/06_train_checkpoint/tsla_{suffix} "
        f"--result-path data/05_train_model_output/tsla_{suffix}"
    )
    print(
        "python3 run.py sim "
        "--market-data-path data/06_input/subset_symbols_TSLA.pkl "
        "--start-time 2026-01-02 "
        "--end-time 2026-03-06 "
        "--run-model test "
        f"--config-path {config_path} "
        f"--trained-agent-path data/05_train_model_output/tsla_{suffix} "
        f"--checkpoint-path data/06_test_checkpoint/tsla_{suffix} "
        f"--result-path data/05_test_model_output/tsla_{suffix}"
    )


def main() -> None:
    base_text = BASE_CONFIG.read_text(encoding="utf-8")
    for top_k, rel_path in TOPK_VARIANTS.items():
        output_path = REPO_ROOT / rel_path
        output_path.write_text(_replace_top_k(base_text, top_k), encoding="utf-8")
        print(f"Wrote {output_path}")
        _print_commands(rel_path, f"top{top_k}")


if __name__ == "__main__":
    main()

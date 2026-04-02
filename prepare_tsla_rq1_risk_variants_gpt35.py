import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = REPO_ROOT / "config/tsla_gpt35_paper_config.toml"

RISK_VARIANTS = {
    "risk_averse": (
        "config/tsla_gpt35_top5_risk_averse_config.toml",
        "Always behave as a risk-averse investor. Prioritize capital preservation. "
        "Give more weight to downside risk, negative news, and uncertainty. "
        "Require stronger evidence before choosing buy.",
    ),
    "risk_seeking": (
        "config/tsla_gpt35_top5_risk_seeking_config.toml",
        "Always behave as a risk-seeking investor. Prioritize upside capture and growth. "
        "Give more weight to momentum, positive news, and asymmetric upside. "
        "Be more willing to buy under uncertainty.",
    ),
    "self_adaptive": (
        "config/tsla_gpt35_top5_self_adaptive_config.toml",
        "Use self-adaptive risk preference. When cumulative return is positive or zero, lean risk-seeking. "
        "When cumulative return is negative, lean risk-averse. "
        "Adjust your behavior dynamically based on current portfolio performance.",
    ),
}


def _replace_scalar(text: str, key: str, value: str) -> str:
    updated, count = re.subn(rf"(?m)^{key}\s*=\s*.+$", f"{key} = {value}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not find {key} in base config.")
    return updated


def _update_chat_block(text: str) -> str:
    if re.search(r"(?m)^temperature\s*=", text):
        return re.sub(r"(?m)^temperature\s*=.*$", "temperature = 0.7", text, count=1)

    updated, count = re.subn(
        r'(?m)^(system_message\s*=\s*".*")\s*$',
        r'\1\ntemperature = 0.7',
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Could not insert temperature line in chat block.")
    return updated


def _replace_character_string(text: str, extra_instruction: str) -> str:
    pattern = r"(?s)(character_string\s*=\s*'''\n)(.*?)(\n''')"
    match = re.search(pattern, text)
    if not match:
        raise ValueError("Could not find character_string block in base config.")
    original = match.group(2).rstrip()
    replacement_body = original + "\n\nRisk profile instruction:\n" + extra_instruction
    return re.sub(pattern, r"\1" + replacement_body + r"\3", text, count=1)


def _print_commands(config_path: str, suffix: str) -> None:
    print(f"\n[{suffix}]")
    print(
        "python3 run.py sim "
        "--market-data-path data/06_input/subset_symbols_TSLA.pkl "
        "--start-time 2025-01-02 "
        "--end-time 2025-12-31 "
        "--run-model train "
        f"--config-path {config_path} "
        f"--checkpoint-path data/06_train_checkpoint/tsla_rq1_gpt35_{suffix} "
        f"--result-path data/05_train_model_output/tsla_rq1_gpt35_{suffix}"
    )
    print(
        "python3 run.py sim "
        "--market-data-path data/06_input/subset_symbols_TSLA.pkl "
        "--start-time 2026-01-02 "
        "--end-time 2026-03-06 "
        "--run-model test "
        f"--config-path {config_path} "
        f"--trained-agent-path data/05_train_model_output/tsla_rq1_gpt35_{suffix} "
        f"--checkpoint-path data/06_test_checkpoint/tsla_rq1_gpt35_{suffix} "
        f"--result-path data/05_test_model_output/tsla_rq1_gpt35_{suffix}"
    )


def main() -> None:
    base_text = BASE_CONFIG.read_text(encoding="utf-8")
    base_text = _update_chat_block(base_text)
    base_text = _replace_scalar(base_text, "top_k", "5")
    base_text = _replace_scalar(base_text, "look_back_window_size", "3")
    for suffix, (rel_path, extra_instruction) in RISK_VARIANTS.items():
        updated = _replace_character_string(base_text, extra_instruction)
        output_path = REPO_ROOT / rel_path
        output_path.write_text(
            f"# Auto-generated from {BASE_CONFIG.name}\n"
            f"# Variant: TSLA GPT-3.5-Turbo temperature=0.7 top_k=5 look_back_window_size=3 {suffix}\n\n"
            f"{updated}",
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
        _print_commands(rel_path, suffix)


if __name__ == "__main__":
    main()

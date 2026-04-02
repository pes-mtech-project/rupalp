import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TICKERS = ["tsla", "amzn", "nflx", "msft", "coin"]
RISK_INSTRUCTION = (
    "Use self-adaptive risk preference. When cumulative return is positive or zero, lean risk-seeking. "
    "When cumulative return is negative, lean risk-averse. "
    "Adjust your behavior dynamically based on current portfolio performance."
)


def _replace_scalar(text: str, key: str, value: str) -> str:
    updated, count = re.subn(rf"(?m)^{key}\s*=\s*.+$", f"{key} = {value}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not find {key} in config.")
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


def _replace_character_string(text: str) -> str:
    pattern = r"(?s)(character_string\s*=\s*'''\n)(.*?)(\n''')"
    match = re.search(pattern, text)
    if not match:
        raise ValueError("Could not find character_string block in config.")
    original = match.group(2).rstrip()
    if "Risk profile instruction:" in original:
        updated_body = re.sub(
            r"(?s)\n\nRisk profile instruction:\n.*$",
            "\n\nRisk profile instruction:\n" + RISK_INSTRUCTION,
            original,
            count=1,
        )
    else:
        updated_body = original + "\n\nRisk profile instruction:\n" + RISK_INSTRUCTION
    return re.sub(pattern, r"\1" + updated_body + r"\3", text, count=1)


def build_config(ticker: str) -> None:
    src = REPO_ROOT / f"config/{ticker}_gpt_config.toml"
    dest = REPO_ROOT / f"config/{ticker}_gpt_rq1_self_adaptive_new_config.toml"
    text = src.read_text(encoding="utf-8")
    text = _update_chat_block(text)
    text = _replace_scalar(text, "top_k", "5")
    text = _replace_scalar(text, "look_back_window_size", "3")
    text = _replace_character_string(text)
    dest.write_text(
        f"# Auto-generated from {src.name}\n"
        "# RQ1 paper-aligned GPT-3.5 self-adaptive variant\n\n"
        f"{text}",
        encoding="utf-8",
    )
    print(f"Wrote {dest}")


if __name__ == "__main__":
    for ticker in TICKERS:
        build_config(ticker)

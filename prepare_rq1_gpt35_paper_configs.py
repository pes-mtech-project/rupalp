import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TICKERS = ["tsla", "amzn", "nflx", "msft", "coin"]


def _update_chat_block(text: str) -> str:
    if re.search(r"(?m)^temperature\s*=", text):
        updated = re.sub(r"(?m)^temperature\s*=.*$", "temperature = 0.7", text, count=1)
    else:
        updated, count = re.subn(
            r'(?m)^(system_message\s*=\s*".*")\s*$',
            r'\1\ntemperature = 0.7',
            text,
            count=1,
        )
        if count != 1:
            raise ValueError("Could not insert temperature line in chat block.")
    return updated


def _replace_scalar(text: str, key: str, value: str) -> str:
    updated, count = re.subn(rf"(?m)^{key}\s*=\s*.+$", f"{key} = {value}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not find {key} in config.")
    return updated


def build_config(src: Path, dest: Path) -> None:
    text = src.read_text(encoding="utf-8")
    text = _update_chat_block(text)
    text = _replace_scalar(text, "top_k", "5")
    text = _replace_scalar(text, "look_back_window_size", "3")
    dest.write_text(
        f"# Auto-generated from {src.name}\n"
        "# RQ1 paper-aligned settings with GPT-3.5 backbone\n\n"
        f"{text}",
        encoding="utf-8",
    )


def main() -> None:
    for ticker in TICKERS:
        src = REPO_ROOT / f"config/{ticker}_gpt_config.toml"
        dest = REPO_ROOT / f"config/{ticker}_gpt_rq1_paper_config.toml"
        build_config(src, dest)
        print(f"Wrote {dest}")


if __name__ == "__main__":
    main()

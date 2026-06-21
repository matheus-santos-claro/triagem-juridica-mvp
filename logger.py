import json
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def save_interaction(payload: dict):

    today = datetime.now().strftime("%Y-%m-%d")

    log_file = LOG_DIR / f"{today}.jsonl"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=False
            )
            + "\n"
        )
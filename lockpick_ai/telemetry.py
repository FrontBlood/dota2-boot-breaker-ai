from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TelemetryWriter:
    def __init__(self, enabled: bool, directory: str | Path = "runs"):
        self.enabled = enabled
        self.handle = None
        self.path: Path | None = None
        self._pending = 0
        if enabled:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = root / f"telemetry_{stamp}.jsonl"
            self.handle = self.path.open("w", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        if self.handle is None:
            return
        self.handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._pending += 1
        if self._pending >= 30 or event.get("click"):
            self.handle.flush()
            self._pending = 0

    def close(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

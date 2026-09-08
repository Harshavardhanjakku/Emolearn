"""
Thread-safe read/write for emotion_log.json.

Concurrent access from the emotion thread (append) and API handlers (read) caused
truncated JSON and WinError 32 when replacing files on Windows.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

FILENAME = "emotion_log.json"

_lock = threading.Lock()


def _reset_file_unlocked() -> None:
    for attempt in range(8):
        try:
            with open(FILENAME, "w", encoding="utf-8") as f:
                f.write("[]")
            return
        except OSError:
            time.sleep(0.04 * (attempt + 1))


def _try_quarantine_corrupt_unlocked(exc: BaseException) -> None:
    ts = int(time.time())
    bak = f"{FILENAME}.broken.{ts}"
    logger.warning("Emotion log invalid (%s); quarantine -> %s", exc, bak)
    try:
        shutil.copy2(FILENAME, bak)
    except OSError as e:
        logger.debug("Backup copy skipped: %s", e)
    _reset_file_unlocked()


def _parse_or_empty(raw: str) -> List[Any]:
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        _try_quarantine_corrupt_unlocked(e)
        return []


def load_entries() -> List[Any]:
    with _lock:
        if not os.path.exists(FILENAME):
            return []
        try:
            with open(FILENAME, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError as e:
            logger.warning("Cannot read emotion log: %s", e)
            return []
        return _parse_or_empty(raw)


def append_entry(entry: Dict[str, Any]) -> None:
    with _lock:
        data: List[Any] = []
        if os.path.exists(FILENAME):
            try:
                with open(FILENAME, "r", encoding="utf-8") as f:
                    raw = f.read()
                data = _parse_or_empty(raw)
            except OSError:
                data = []

        data.append(entry)

        for attempt in range(8):
            try:
                with open(FILENAME, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                logger.info("[Combined Emotion] %s", entry)
                return
            except OSError as e:
                logger.warning("emotion_log.json write retry %s/8: %s", attempt + 1, e)
                time.sleep(0.05 * (attempt + 1))
        logger.error("Could not persist emotion_log.json after retries.")

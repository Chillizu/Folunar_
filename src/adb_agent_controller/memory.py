from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class MemoryEntry:
    app: str
    note: str
    timestamp: str

    @classmethod
    def create(cls, app: str, note: str) -> "MemoryEntry":
        return cls(app=app, note=note, timestamp=datetime.utcnow().isoformat())


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(self) -> List[MemoryEntry]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [MemoryEntry(**entry) for entry in data]

    def save(self, entries: Iterable[MemoryEntry]) -> None:
        payload = [asdict(entry) for entry in entries]
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def add(self, entry: MemoryEntry) -> None:
        entries = self.load()
        entries.append(entry)
        self.save(entries)

    def list(self, app: Optional[str] = None) -> List[MemoryEntry]:
        entries = self.load()
        if app:
            entries = [entry for entry in entries if entry.app.lower() == app.lower()]
        return entries

"""Flat-file persistence.

A household's inventory is small and a hackathon should not need a database.
Everything lives in one JSON file that a human can open and read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    Alert,
    Action,
    Hazard,
    InventoryItem,
    SilenceEntry,
    SweepReport,
    to_jsonable,
)

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "state.json"


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DEFAULT_PATH)
        self._data: dict[str, Any] = {
            "inventory": [],
            "alerts": [],
            "silence_log": [],
            "sweeps": [],
            "seen_recalls": [],
        }
        if self.path.exists():
            self._data.update(json.loads(self.path.read_text(encoding="utf-8")))

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # -- inventory ---------------------------------------------------------

    @property
    def inventory(self) -> list[InventoryItem]:
        return [InventoryItem(**row) for row in self._data["inventory"]]

    def add_items(self, items: list[InventoryItem]) -> list[InventoryItem]:
        existing = {row["id"] for row in self._data["inventory"]}
        added = []
        for item in items:
            if item.id in existing:
                continue
            self._data["inventory"].append(to_jsonable(item))
            existing.add(item.id)
            added.append(item)
        return added

    def remove_item(self, item_id: str) -> bool:
        before = len(self._data["inventory"])
        self._data["inventory"] = [
            row for row in self._data["inventory"] if row["id"] != item_id
        ]
        return len(self._data["inventory"]) < before

    def item(self, item_id: str) -> InventoryItem | None:
        for row in self._data["inventory"]:
            if row["id"] == item_id:
                return InventoryItem(**row)
        return None

    # -- alerts ------------------------------------------------------------

    @property
    def alerts(self) -> list[Alert]:
        out = []
        for row in self._data["alerts"]:
            row = dict(row)
            row["action"] = Action(row["action"])
            row["hazard"] = Hazard(row["hazard"])
            out.append(Alert(**row))
        return out

    def add_alerts(self, alerts: list[Alert]) -> list[Alert]:
        known = {(a["item_id"], a["recall_id"]) for a in self._data["alerts"]}
        fresh = [a for a in alerts if (a.item_id, a.recall_id) not in known]
        self._data["alerts"].extend(to_jsonable(a) for a in fresh)
        return fresh

    def acknowledge(self, item_id: str, recall_id: str) -> bool:
        for row in self._data["alerts"]:
            if row["item_id"] == item_id and row["recall_id"] == recall_id:
                row["acknowledged"] = True
                return True
        return False

    # -- silence log -------------------------------------------------------

    @property
    def silence_log(self) -> list[SilenceEntry]:
        return [SilenceEntry(**row) for row in self._data["silence_log"]]

    def add_silence(self, entries: list[SilenceEntry], cap: int = 500) -> None:
        self._data["silence_log"].extend(to_jsonable(e) for e in entries)
        self._data["silence_log"] = self._data["silence_log"][-cap:]

    # -- sweeps ------------------------------------------------------------

    @property
    def sweeps(self) -> list[SweepReport]:
        return [SweepReport(**row) for row in self._data["sweeps"]]

    def add_sweep(self, report: SweepReport) -> None:
        self._data["sweeps"].append(to_jsonable(report))

    @property
    def total_recalls_seen(self) -> int:
        return sum(s["recalls_seen"] for s in self._data["sweeps"])

    def mark_seen(self, recall_ids: list[str]) -> None:
        seen = set(self._data["seen_recalls"]) | set(recall_ids)
        self._data["seen_recalls"] = sorted(seen)

    @property
    def seen_recalls(self) -> set[str]:
        return set(self._data["seen_recalls"])

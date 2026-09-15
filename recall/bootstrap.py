"""Startup seeding for hosted demos.

A public demo has to show something the moment someone opens the link, and its
filesystem is usually thrown away on every deploy. So when RECALL_DEMO=1 and the
store is empty, load the demo household and run one offline sweep before the
first request arrives.

Deliberately a no-op when a real store already exists: this must never touch
somebody's actual inventory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .agents import ReplayBrain
from .feeds import load_snapshots
from .models import InventoryItem
from .pipeline import run_sweep
from .store import Store

DEMO_INVENTORY = Path(__file__).resolve().parent.parent / "data" / "demo_inventory.json"


def ensure_demo_state() -> str | None:
    """Seed and sweep once, if this is a demo deployment with nothing in it."""
    if os.environ.get("RECALL_DEMO") != "1":
        return None

    store = Store()
    if store.inventory:
        return "already populated"

    rows = json.loads(DEMO_INVENTORY.read_text(encoding="utf-8"))
    store.add_items([InventoryItem(**row) for row in rows])
    store.save()

    recalls = load_snapshots()
    if not recalls:
        return "seeded inventory, but no recall corpus on disk"

    report = run_sweep(store, recalls, ReplayBrain())
    return (
        f"seeded {len(store.inventory)} items, read {report.recalls_seen} recalls, "
        f"raised {report.alerts_raised} alerts, dismissed {report.silenced}"
    )

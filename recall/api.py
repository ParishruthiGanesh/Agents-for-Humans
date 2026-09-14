"""HTTP interface.

Endpoints are deliberately synchronous: Starlette runs them in a threadpool, and
the agent calls underneath are blocking. That keeps the code readable and avoids
pretending a model call is cheap.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import build_brain
from .feeds import load_snapshots
from .models import Action, InventoryItem, to_jsonable
from .pipeline import run_sweep
from .store import Store
from .tools import bind_store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
OFFLINE = os.environ.get("RECALL_OFFLINE") == "1"

app = FastAPI(title="Recall", description="An agent that watches your possessions.")


def get_store() -> Store:
    store = Store()
    bind_store(store)
    return store


class PasteRequest(BaseModel):
    text: str


class AskRequest(BaseModel):
    question: str


class AckRequest(BaseModel):
    item_id: str
    recall_id: str


@app.get("/api/state")
def state() -> dict:
    store = get_store()
    recalls = {r.id: r for r in load_snapshots()}
    alerts = []
    for alert in store.alerts:
        item = store.item(alert.item_id)
        recall = recalls.get(alert.recall_id)
        alerts.append(
            {
                **to_jsonable(alert),
                "item_name": item.name if item else alert.item_id,
                "item_description": item.describe() if item else "",
                "source": recall.source.value if recall else None,
                "url": recall.url if recall else None,
                "recall_title": recall.title if recall else alert.recall_id,
            }
        )
    alerts.sort(
        key=lambda a: (a["acknowledged"], a["action"] != Action.NOTIFY_NOW.value)
    )

    sweeps = store.sweeps
    return {
        "inventory": [to_jsonable(i) for i in store.inventory],
        "alerts": alerts,
        "silence_log": [to_jsonable(e) for e in reversed(store.silence_log[-60:])],
        "stats": {
            "items": len(store.inventory),
            "recalls_read": store.total_recalls_seen,
            "sweeps": len(sweeps),
            "open_alerts": sum(1 for a in store.alerts if not a.acknowledged),
            "needs_you": sum(
                1
                for a in store.alerts
                if not a.acknowledged and a.action is Action.NOTIFY_NOW
            ),
            "silenced": len(store.silence_log),
            "last_sweep": sweeps[-1].finished_at if sweeps else None,
            "corpus_size": len(recalls),
        },
        "offline": OFFLINE,
    }


@app.post("/api/sweep")
def sweep(rescan: bool = False) -> dict:
    store = get_store()
    if not store.inventory:
        raise HTTPException(400, "Add something you own before sweeping.")
    recalls = load_snapshots()
    if not recalls:
        raise HTTPException(400, "No recall corpus on disk. Run `recall fetch` first.")
    report = run_sweep(store, recalls, build_brain(offline=OFFLINE), skip_seen=not rescan)
    return to_jsonable(report)


@app.post("/api/items/paste")
def add_from_text(request: PasteRequest) -> dict:
    store = get_store()
    extracted = build_brain(offline=OFFLINE).extract_items(request.text)
    items = [
        InventoryItem(
            id=InventoryItem.make_id(e.brand, e.name, e.purchased_on),
            name=e.name,
            brand=e.brand,
            model=e.model,
            category=e.category,
            purchased_on=e.purchased_on,
            quantity=e.quantity,
            source_note="extracted from pasted text",
        )
        for e in extracted.items
    ]
    added = store.add_items(items)
    store.save()
    return {
        "added": [to_jsonable(i) for i in added],
        "skipped": extracted.skipped,
    }


@app.delete("/api/items/{item_id}")
def remove_item(item_id: str) -> dict:
    store = get_store()
    if not store.remove_item(item_id):
        raise HTTPException(404, "No such item.")
    store.save()
    return {"removed": item_id}


@app.post("/api/alerts/ack")
def acknowledge(request: AckRequest) -> dict:
    store = get_store()
    if not store.acknowledge(request.item_id, request.recall_id):
        raise HTTPException(404, "No such alert.")
    store.save()
    return {"acknowledged": True}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    if OFFLINE:
        raise HTTPException(
            503, "Asking needs a live model. Restart without RECALL_OFFLINE=1."
        )
    from .tools import build_assistant

    get_store()
    return {"answer": str(build_assistant()(request.question))}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

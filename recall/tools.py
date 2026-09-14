"""Strands tools, and the one agent that is allowed to talk.

The pipeline agents are silent specialists. This is the conversational surface:
ask it a question about your own possessions and it calls these functions to
answer. Each is an ordinary Python function; the @tool decorator is what makes
it callable by the model.
"""

from __future__ import annotations

from strands import Agent, tool

from .blocking import score_pair
from .models import Action
from .store import Store

_store: Store | None = None


def bind_store(store: Store) -> None:
    """Point the tools at a store. Called once at startup."""
    global _store
    _store = store


def _require_store() -> Store:
    if _store is None:
        raise RuntimeError("call bind_store() before using the tools")
    return _store


@tool
def search_inventory(query: str) -> str:
    """Find things the person owns whose name, brand or category matches a query.

    Args:
        query: A word or phrase, for example "car seat", "Graco", or "kitchen".
    """
    store = _require_store()
    needle = query.lower().strip()
    hits = [
        item
        for item in store.inventory
        if needle in " ".join(
            filter(None, [item.name, item.brand, item.model, item.category])
        ).lower()
    ]
    if not hits:
        return f"Nothing in the inventory matches {query!r}."
    return "\n".join(f"- {item.describe()} [{item.id}]" for item in hits[:25])


@tool
def list_open_alerts() -> str:
    """List recall alerts that are still waiting for the person to act."""
    store = _require_store()
    open_alerts = [a for a in store.alerts if not a.acknowledged]
    if not open_alerts:
        return "No open alerts. Everything checked so far is clear."
    lines = []
    for alert in open_alerts:
        item = store.item(alert.item_id)
        urgency = "needs you now" if alert.action is Action.NOTIFY_NOW else "can wait"
        lines.append(
            f"- {alert.headline} ({urgency}; {alert.hazard.value} hazard)\n"
            f"  item: {item.name if item else alert.item_id}\n"
            f"  what to do: {alert.what_to_do}"
        )
    return "\n".join(lines)


@tool
def why_was_this_ignored(recall_query: str) -> str:
    """Explain why a particular recall was checked but never shown.

    Args:
        recall_query: Part of the recall's title or its id.
    """
    store = _require_store()
    needle = recall_query.lower().strip()
    hits = [
        entry
        for entry in store.silence_log
        if needle in entry.recall_title.lower() or needle in entry.recall_id.lower()
    ]
    if not hits:
        return f"No silence-log entry matches {recall_query!r}."
    return "\n".join(
        f"- {e.recall_title}\n  dismissed at the {e.stage} stage: {e.reason}"
        for e in hits[:10]
    )


@tool
def coverage_summary() -> str:
    """Report how much watching has been done: recalls read, alerts raised, sweeps run."""
    store = _require_store()
    sweeps = store.sweeps
    alerts = store.alerts
    return (
        f"{len(store.inventory)} items watched. "
        f"{store.total_recalls_seen} recalls read across {len(sweeps)} sweeps. "
        f"{len(alerts)} alerts raised, "
        f"{sum(1 for a in alerts if not a.acknowledged)} still open. "
        f"{len(store.silence_log)} recalls checked and deliberately not shown."
    )


@tool
def closest_recall_for_item(item_id: str) -> str:
    """Show the nearest near-miss recalls for one owned item, with scores.

    Useful for answering "are you sure my X is fine?".

    Args:
        item_id: The inventory id, such as itm_1a2b3c4d5e.
    """
    from .feeds import load_snapshots

    store = _require_store()
    item = store.item(item_id)
    if not item:
        return f"No inventory item with id {item_id!r}."

    scored = []
    for recall in load_snapshots():
        score, reasons = score_pair(item, recall)
        if score > 0:
            scored.append((score, recall, reasons))
    scored.sort(key=lambda row: row[0], reverse=True)
    if not scored:
        return f"Nothing in the current recall corpus resembles {item.name}."
    return "\n".join(
        f"- {recall.title} (similarity {score:.2f}): {'; '.join(reasons[:2])}"
        for score, recall, reasons in scored[:5]
    )


ASSISTANT_PROMPT = """You answer questions about one household's possessions and
the recalls that have been checked against them.

Always use the tools rather than guessing; you have no knowledge of this
person's belongings except what the tools return. Be brief and concrete. If
something is a safety risk, lead with what to do about it, not with reassurance.
"""


def build_assistant(model=None) -> Agent:
    """The conversational agent, wired to the tools above."""
    if model is None:
        from .agents import DEFAULT_MODEL_ID, DEFAULT_REGION
        from strands.models import BedrockModel

        model = BedrockModel(
            model_id=DEFAULT_MODEL_ID, region_name=DEFAULT_REGION, temperature=0.3
        )
    return Agent(
        model=model,
        system_prompt=ASSISTANT_PROMPT,
        tools=[
            search_inventory,
            list_open_alerts,
            why_was_this_ignored,
            coverage_summary,
            closest_recall_for_item,
        ],
    )

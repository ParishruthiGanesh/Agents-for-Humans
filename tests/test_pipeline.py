"""End-to-end sweep tests, driven by the replay brain.

No credentials and no network: the recorded judgements in data/replay stand in
for the model, so the orchestration itself is what is under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from recall.agents import ReplayBrain
from recall.feeds import load_snapshots
from recall.models import Action, InventoryItem
from recall.pipeline import run_sweep
from recall.store import Store

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def store(tmp_path) -> Store:
    store = Store(tmp_path / "state.json")
    import json

    rows = json.loads((ROOT / "data" / "demo_inventory.json").read_text())
    store.add_items([InventoryItem(**row) for row in rows])
    store.save()
    return store


@pytest.fixture
def corpus():
    return load_snapshots(ROOT / "data" / "snapshots")


def test_a_sweep_stays_mostly_silent(store, corpus):
    report = run_sweep(store, corpus, ReplayBrain())
    assert report.recalls_seen == len(corpus)
    # The whole premise: far more read and dismissed than surfaced.
    assert report.silenced > report.alerts_raised * 2


def test_the_matcher_kills_what_the_filter_could_not(store, corpus):
    run_sweep(store, corpus, ReplayBrain())
    reasons = {e.stage for e in store.silence_log}
    assert "matcher" in reasons, "near-misses should be dismissed by the matcher"
    matcher_calls = [e for e in store.silence_log if e.stage == "matcher"]
    titles = " ".join(e.recall_title for e in matcher_calls)
    assert "MALM" in titles or "BlendJet" in titles


def test_a_genuine_hazard_surfaces_with_something_to_do(store, corpus):
    run_sweep(store, corpus, ReplayBrain())
    urgent = [a for a in store.alerts if a.action is Action.NOTIFY_NOW]
    assert urgent, "the seeded household owns recalled products"
    for alert in urgent:
        assert alert.what_to_do.strip()
        assert alert.headline.strip()


def test_uncertainty_surfaces_rather_than_guessing(store, corpus):
    run_sweep(store, corpus, ReplayBrain())
    # The Graco seat can only be resolved by reading a label, so the agent must
    # ask rather than decide either way.
    asks = [a for a in store.alerts if "Check" in a.headline or "check" in a.what_to_do]
    assert asks, "a pair needing owner input should be raised as a question"


def test_a_second_sweep_does_not_repeat_itself(store, corpus):
    first = run_sweep(store, corpus, ReplayBrain())
    second = run_sweep(store, corpus, ReplayBrain())
    assert first.alerts_raised > 0
    assert second.recalls_seen == 0
    assert second.alerts_raised == 0


def test_rescanning_re_reads_but_does_not_duplicate_alerts(store, corpus):
    run_sweep(store, corpus, ReplayBrain())
    before = len(store.alerts)
    again = run_sweep(store, corpus, ReplayBrain(), skip_seen=False)
    assert again.recalls_seen == len(corpus)
    assert len(store.alerts) == before, "the same alert must not be raised twice"

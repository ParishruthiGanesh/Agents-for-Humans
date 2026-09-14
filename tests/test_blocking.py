"""Tests for the deterministic layer.

The cheap filter decides what the model never sees, so its mistakes are the
expensive kind: a false negative here is a recall that silently never reaches
anyone. These cases are drawn from the demo corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recall.blocking import generate_candidates, normalise_brand, score_pair, tokenise
from recall.models import InventoryItem, RecallNotice, Source


def item(**kw) -> InventoryItem:
    base = dict(id="itm_test", name="thing", brand=None, model=None,
                category=None, purchased_on=None)
    base.update(kw)
    return InventoryItem(**base)


def notice(**kw) -> RecallNotice:
    base = dict(id="r1", source=Source.CPSC, title="t", description="d")
    base.update(kw)
    return RecallNotice(**base)


class TestNormalisation:
    def test_trading_names_collapse_to_one_brand(self):
        assert normalise_brand("Graco Children's Products") == "graco"
        assert normalise_brand("GRACO") == "graco"

    def test_corporate_suffixes_are_dropped(self):
        assert normalise_brand("Future Motion, Inc.") == "future motion"

    def test_none_is_safe(self):
        assert normalise_brand(None) == ""

    def test_stopwords_leave_the_signal(self):
        assert tokenise("The Blender for Home Use") == {"blender", "home", "use"}


class TestScoring:
    def test_named_brand_and_model_scores_high(self):
        score, reasons = score_pair(
            item(name="BlendJet 2 portable blender", brand="BlendJet", category="blender"),
            notice(title="BlendJet 2 blenders recalled", brands=["BlendJet"],
                   models=["BlendJet 2"], category="blender"),
        )
        assert score > 0.8
        assert any("brand" in r for r in reasons)

    def test_purchase_before_the_window_is_disqualifying(self):
        score, reasons = score_pair(
            item(name="cordless drill", brand="Ryobi", purchased_on="2020-08-15"),
            notice(title="Ryobi drills recalled", brands=["Ryobi"],
                   sold_from="2022-01-01", sold_to="2023-04-30"),
        )
        assert score == 0.0
        assert any("outside the affected window" in r for r in reasons)

    def test_receipt_dates_get_some_slack(self):
        # Two months before the stated window still survives; receipts and sale
        # windows rarely agree exactly.
        score, _ = score_pair(
            item(name="blender", brand="BlendJet", purchased_on="2021-01-15"),
            notice(title="BlendJet blenders", brands=["BlendJet"],
                   sold_from="2021-03-01", sold_to="2024-02-29"),
        )
        assert score > 0.0

    def test_same_category_different_brand_still_survives_for_the_model(self):
        # The cheap filter must not be the thing that rules this out: deciding
        # a Ninja is not a BlendJet is a judgement, not a rule.
        score, _ = score_pair(
            item(name="Professional BN701 blender", brand="Ninja", category="blender"),
            notice(title="BlendJet 2 blenders recalled", brands=["BlendJet"],
                   category="blender"),
        )
        assert score > 0.0

    def test_unrelated_things_score_nothing(self):
        score, _ = score_pair(
            item(name="WH-1000XM4 headphones", brand="Sony", category="headphones"),
            notice(title="Boppy newborn loungers recalled", brands=["Boppy"],
                   category="infant product"),
        )
        assert score < 0.3


class TestCandidateGeneration:
    def test_unmatched_recalls_land_in_the_silence_log(self):
        items = [item(id="a", name="headphones", brand="Sony", category="headphones")]
        recalls = [
            notice(id="r1", title="Boppy loungers", brands=["Boppy"]),
            notice(id="r2", title="Peloton treadmills", brands=["Peloton"]),
        ]
        candidates, silenced = generate_candidates(items, recalls)
        assert candidates == []
        assert {s.recall_id for s in silenced} == {"r1", "r2"}
        assert all(s.stage == "blocking" for s in silenced)

    def test_disposed_items_are_not_watched(self):
        items = [item(id="a", name="BlendJet 2", brand="BlendJet", disposed=True)]
        recalls = [notice(id="r1", title="BlendJet 2 recalled", brands=["BlendJet"])]
        candidates, silenced = generate_candidates(items, recalls)
        assert candidates == []
        assert len(silenced) == 1

    def test_candidates_come_back_strongest_first(self):
        items = [
            item(id="strong", name="BlendJet 2 blender", brand="BlendJet", category="blender"),
            item(id="weak", name="Ninja blender", brand="Ninja", category="blender"),
        ]
        recalls = [notice(id="r1", title="BlendJet 2 blenders recalled",
                          brands=["BlendJet"], models=["BlendJet 2"], category="blender")]
        candidates, _ = generate_candidates(items, recalls)
        # Both survive: the Ninja is the model's call to make, not the filter's.
        assert [c.item_id for c in candidates] == ["strong", "weak"]
        assert candidates[0].score > candidates[1].score

"""The Strands layer.

Three specialists, each with one job and a typed output:

  ReceiptExtractor  messy receipt or order email  ->  inventory items
  RecallMatcher     one (item, recall) pair       ->  match / no match / need info
  RecallTriage      a confirmed match             ->  severity, action, draft message

Splitting them matters. Matching is a narrow evidence question asked hundreds of
times per sweep; triage is a judgement call about whether a human's attention is
worth spending, asked only a handful of times. Different jobs, different prompts,
different costs.

A fourth agent in tools.py is conversational and tool-calling: it answers
questions about your own inventory using the functions in that module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from .models import (
    Action,
    Candidate,
    Hazard,
    InventoryItem,
    MatchResult,
    RecallNotice,
    Verdict,
)

DEFAULT_MODEL_ID = os.environ.get(
    "RECALL_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-west-2")
REPLAY_DIR = Path(__file__).resolve().parent.parent / "data" / "replay"


# ---------------------------------------------------------------------------
# Typed outputs
# ---------------------------------------------------------------------------

class ExtractedItem(BaseModel):
    name: str = Field(description="Plain product name, no marketing copy")
    brand: str | None = Field(default=None, description="Manufacturer or retail brand")
    model: str | None = Field(default=None, description="Model or part number if stated")
    category: str | None = Field(
        default=None,
        description="Broad category, e.g. 'car seat', 'blender', 'infant formula'",
    )
    purchased_on: str | None = Field(default=None, description="ISO date YYYY-MM-DD")
    quantity: int = Field(default=1)


class ExtractedItems(BaseModel):
    items: list[ExtractedItem]
    skipped: list[str] = Field(
        default_factory=list,
        description="Lines deliberately not treated as durable possessions",
    )


class MatchJudgement(BaseModel):
    verdict: str = Field(description="One of: match, no_match, need_info")
    confidence: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="One or two sentences a non-expert can follow")
    missing_info: str | None = Field(
        default=None,
        description="If need_info, exactly what the owner must check (e.g. a serial number)",
    )


class TriageDecision(BaseModel):
    hazard: str = Field(description="One of: death, injury, illness, property, labelling, unknown")
    action: str = Field(description="One of: notify_now, digest, archive")
    headline: str = Field(description="Under 90 characters, plain language, no alarm words")
    what_happened: str = Field(description="Two sentences on the actual defect and risk")
    what_to_do: str = Field(description="Concrete next step, including how to claim a remedy")
    draft_message: str | None = Field(
        default=None, description="Ready-to-send message to the manufacturer, if one helps"
    )
    deadline: str | None = Field(default=None, description="ISO date if the remedy expires")


# ---------------------------------------------------------------------------
# Brain: the seam between real model calls and deterministic replay
# ---------------------------------------------------------------------------

class Brain(Protocol):
    """What the pipeline needs from a reasoning engine."""

    def extract_items(self, text: str) -> ExtractedItems: ...

    def judge_match(
        self, item: InventoryItem, recall: RecallNotice, candidate: Candidate
    ) -> MatchJudgement: ...

    def triage(
        self, item: InventoryItem, recall: RecallNotice, match: MatchResult
    ) -> TriageDecision: ...


EXTRACTOR_PROMPT = """You turn messy purchase records into a list of durable possessions.

Input is whatever a person pasted: an order confirmation email, a photographed
receipt transcribed to text, a bank statement line, or a list they typed.

Rules:
- Keep only durable goods that could plausibly ever be recalled: appliances,
  electronics, vehicles, car seats, cribs, tools, furniture, medical devices,
  supplements, packaged food and formula.
- Discard services, subscriptions, digital goods, restaurant meals, fuel,
  postage, taxes and shipping lines. Put a short note for each in `skipped`.
- Never invent a model number, a brand or a date. Leave the field null if the
  text does not state it.
- Dates must be ISO YYYY-MM-DD. If only a month is given, use the first day.
"""

MATCHER_PROMPT = """You decide whether a recall notice covers a specific item someone owns.

You are the careful step between a crude keyword filter and interrupting a human
being. A false alarm costs their trust; a miss can cost far more. Weigh the
evidence actually present rather than the surface similarity of words.

Consider:
- Brand names differ between receipts and notices ("Graco Children's Products"
  and "Graco" are the same company; "Kirkland" is not "Costco Wholesale" as a
  manufacturer).
- Model families matter. A recall of the "4Ever DLX" does not necessarily cover
  every "4Ever".
- Purchase dates outside the stated affected window usually rule a match out,
  but receipts and sale windows are both imprecise.
- If the only thing standing between you and certainty is something the owner
  could look up in thirty seconds, such as a serial number, a lot code or a
  manufacture date on a sticker, answer `need_info` and say exactly what to check.

Answer `match` only when a reasonable safety-conscious person would act on it.
"""

TRIAGE_PROMPT = """You decide whether a confirmed recall is worth interrupting someone's day.

The agent this belongs to is silent by design. It reads thousands of recalls and
shows almost none of them. Your job is to protect that silence while never
sitting on something that could hurt someone.

- notify_now: a plausible risk of death, injury or serious illness, or a remedy
  with a deadline that is close.
- digest: real and worth knowing, but it can wait for the weekly summary.
  Labelling errors on something already consumed, cosmetic defects, remedies
  with no deadline.
- archive: technically a match but there is nothing for this person to do.

Write for a worried non-expert. No jargon, no hedging, no exclamation marks.
`what_to_do` must be a concrete next step, not "contact the manufacturer" alone:
name the remedy on offer and how to claim it. If a written message would help
them get a refund or replacement, draft it in `draft_message` in the first
person, brief and factual, with a placeholder for anything you do not know.
"""


class StrandsBrain:
    """Real reasoning, via Strands agents over Amazon Bedrock."""

    def __init__(self, model_id: str | None = None, region: str | None = None) -> None:
        from strands import Agent
        from strands.models import BedrockModel

        self.model = BedrockModel(
            model_id=model_id or DEFAULT_MODEL_ID,
            region_name=region or DEFAULT_REGION,
            temperature=0.2,
        )
        self.extractor = Agent(model=self.model, system_prompt=EXTRACTOR_PROMPT)
        self.matcher = Agent(model=self.model, system_prompt=MATCHER_PROMPT)
        self.triager = Agent(model=self.model, system_prompt=TRIAGE_PROMPT)

    def extract_items(self, text: str) -> ExtractedItems:
        return self.extractor.structured_output(
            ExtractedItems, f"Extract possessions from this purchase record:\n\n{text}"
        )

    def judge_match(
        self, item: InventoryItem, recall: RecallNotice, candidate: Candidate
    ) -> MatchJudgement:
        prompt = f"""ITEM THE PERSON OWNS
{item.describe()}
category: {item.category or 'unknown'}

RECALL NOTICE ({recall.source.value}, published {recall.published_on or 'unknown'})
title: {recall.title}
brands named: {', '.join(recall.brands) or 'none listed'}
models named: {', '.join(recall.models) or 'none listed'}
affected sale window: {recall.sold_from or '?'} to {recall.sold_to or '?'}
description: {recall.description}
hazard: {recall.hazard_text or 'not stated'}

WHY THE CHEAP FILTER KEPT THIS PAIR (score {candidate.score})
- """ + "\n- ".join(candidate.reasons or ["no specific signal"]) + """

Does this recall cover the item this person owns?"""
        return self.matcher.structured_output(MatchJudgement, prompt)

    def triage(
        self, item: InventoryItem, recall: RecallNotice, match: MatchResult
    ) -> TriageDecision:
        prompt = f"""CONFIRMED MATCH (confidence {match.confidence})
{match.reasoning}

THE PERSON OWNS
{item.describe()}

THE RECALL
title: {recall.title}
description: {recall.description}
hazard: {recall.hazard_text or 'not stated'}
remedy offered: {recall.remedy_text or 'not stated'}
who to contact: {recall.contact or 'not stated'}
more information: {recall.url or 'not stated'}

Decide how this should reach them, and write it."""
        return self.triager.structured_output(TriageDecision, prompt)


class ReplayBrain:
    """Deterministic stand-in used by tests and the offline demo.

    Reads judgements recorded from a real run so the pipeline, API and UI can be
    exercised end to end with no credentials and no network. Anything it has not
    seen is answered conservatively rather than guessed at.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory or REPLAY_DIR)
        self._matches = self._load("matches.json")
        self._triage = self._load("triage.json")
        self._extractions = self._load("extractions.json")

    def _load(self, name: str) -> dict:
        path = self.dir / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def extract_items(self, text: str) -> ExtractedItems:
        for key, payload in self._extractions.items():
            if key.lower() in text.lower():
                return ExtractedItems(**payload)
        return ExtractedItems(items=[], skipped=["replay brain has no recording for this text"])

    def judge_match(
        self, item: InventoryItem, recall: RecallNotice, candidate: Candidate
    ) -> MatchJudgement:
        recorded = self._matches.get(f"{item.id}|{recall.id}")
        if recorded:
            return MatchJudgement(**recorded)
        return MatchJudgement(
            verdict=Verdict.NEED_INFO.value,
            confidence=0.4,
            reasoning=(
                "No recorded judgement for this pair; the cheap filter kept it on "
                + (candidate.reasons[0] if candidate.reasons else "a weak signal")
                + "."
            ),
            missing_info="Run with a live model to adjudicate this pair.",
        )

    def triage(
        self, item: InventoryItem, recall: RecallNotice, match: MatchResult
    ) -> TriageDecision:
        recorded = self._triage.get(f"{item.id}|{recall.id}")
        if recorded:
            return TriageDecision(**recorded)
        return TriageDecision(
            hazard=Hazard.UNKNOWN.value,
            action=Action.DIGEST.value,
            headline=f"Possible recall affecting your {item.name}",
            what_happened=recall.hazard_text or recall.description[:200],
            what_to_do=f"Check {recall.url or 'the manufacturer site'} for the remedy.",
        )


def build_brain(offline: bool = False) -> Brain:
    """Pick a reasoning engine. Offline mode never touches the network."""
    if offline or os.environ.get("RECALL_OFFLINE") == "1":
        return ReplayBrain()
    return StrandsBrain()

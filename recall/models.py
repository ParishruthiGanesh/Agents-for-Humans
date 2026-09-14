"""Core domain types for Recall.

Deliberately plain dataclasses: every stage of the pipeline reads and writes
these, and they serialise to JSON without ceremony so the whole run is
inspectable on disk.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class Source(str, Enum):
    """Public recall feeds Recall watches."""

    CPSC = "cpsc"          # Consumer Product Safety Commission
    NHTSA = "nhtsa"        # vehicles, tyres, car seats
    FDA_FOOD = "fda_food"  # food enforcement reports
    FDA_DRUG = "fda_drug"  # drug enforcement reports
    FDA_DEVICE = "fda_device"


class Hazard(str, Enum):
    """Severity of what the recall says can happen to you."""

    DEATH = "death"
    INJURY = "injury"
    ILLNESS = "illness"
    PROPERTY = "property"
    LABELLING = "labelling"
    UNKNOWN = "unknown"

    @property
    def rank(self) -> int:
        return {
            Hazard.DEATH: 5,
            Hazard.INJURY: 4,
            Hazard.ILLNESS: 3,
            Hazard.PROPERTY: 2,
            Hazard.LABELLING: 1,
            Hazard.UNKNOWN: 0,
        }[self]


class Verdict(str, Enum):
    """What the matcher concluded about one (item, recall) pair."""

    MATCH = "match"
    NO_MATCH = "no_match"
    NEED_INFO = "need_info"   # can't decide without something only the owner knows


class Action(str, Enum):
    """What triage decided to do about a confirmed match."""

    NOTIFY_NOW = "notify_now"   # interrupt the person today
    DIGEST = "digest"           # hold for the weekly summary
    ARCHIVE = "archive"         # real match, no action needed


@dataclass
class InventoryItem:
    """Something the person owns."""

    id: str
    name: str
    brand: str | None = None
    model: str | None = None
    category: str | None = None
    purchased_on: str | None = None     # ISO date
    serial: str | None = None
    lot_code: str | None = None
    quantity: int = 1
    source_note: str | None = None      # where we learned about it
    disposed: bool = False              # sold, binned, given away

    @staticmethod
    def make_id(brand: str | None, name: str, purchased_on: str | None) -> str:
        raw = f"{(brand or '').lower()}|{name.lower()}|{purchased_on or ''}"
        return "itm_" + hashlib.sha1(raw.encode()).hexdigest()[:10]

    @property
    def purchase_date(self) -> date | None:
        return _parse_date(self.purchased_on)

    def describe(self) -> str:
        # Receipts often repeat the brand and model inside the product name, so
        # only prepend or append what is not already there.
        line = self.name
        lowered = line.lower()
        if self.brand and self.brand.lower() not in lowered:
            line = f"{self.brand} {line}"
        if self.model and self.model.lower() not in lowered:
            line = f"{line} ({self.model})"
        if self.purchased_on:
            line += f" (bought {self.purchased_on})"
        if self.serial:
            line += f" serial {self.serial}"
        if self.lot_code:
            line += f" lot {self.lot_code}"
        return line


@dataclass
class RecallNotice:
    """One recall as published by a public feed."""

    id: str
    source: Source
    title: str
    description: str
    published_on: str | None = None
    brands: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    category: str | None = None
    hazard_text: str | None = None
    remedy_text: str | None = None
    contact: str | None = None
    url: str | None = None
    units: str | None = None
    sold_from: str | None = None   # ISO date, start of the affected sale window
    sold_to: str | None = None     # ISO date, end of it

    def searchable(self) -> str:
        return " ".join(
            filter(
                None,
                [
                    self.title,
                    self.description,
                    self.hazard_text,
                    " ".join(self.brands),
                    " ".join(self.models),
                    self.category,
                ],
            )
        )

    @property
    def sold_window(self) -> tuple[date | None, date | None]:
        return _parse_date(self.sold_from), _parse_date(self.sold_to)


@dataclass
class Candidate:
    """A (item, recall) pair the cheap deterministic pass could not rule out."""

    item_id: str
    recall_id: str
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class MatchResult:
    """The matcher agent's judgement on one candidate pair."""

    item_id: str
    recall_id: str
    verdict: Verdict
    confidence: float
    reasoning: str
    missing_info: str | None = None   # what we'd need to be sure, when NEED_INFO


@dataclass
class Alert:
    """A confirmed match, triaged and ready to show a human."""

    item_id: str
    recall_id: str
    action: Action
    hazard: Hazard
    headline: str
    what_happened: str
    what_to_do: str
    draft_message: str | None = None
    deadline: str | None = None
    confidence: float = 0.0
    reasoning: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    acknowledged: bool = False


@dataclass
class SilenceEntry:
    """A recall that was checked and deliberately not shown.

    The silence log is the product's credibility: it is the evidence that the
    agent read everything and chose to stay quiet, rather than simply missing
    things.
    """

    recall_id: str
    recall_title: str
    reason: str
    stage: str          # "blocking" or "matcher" or "triage"
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class SweepReport:
    """Summary of one full pass over the feeds."""

    started_at: str
    finished_at: str
    recalls_seen: int
    candidates: int
    adjudicated: int
    alerts_raised: int
    notify_now: int
    silenced: int
    sources: list[str] = field(default_factory=list)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value.strip()[: len(fmt) + 2], fmt).date()
        except ValueError:
            continue
    return None


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses and enums into JSON-safe values."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj

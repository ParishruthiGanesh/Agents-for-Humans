"""The sweep: from public feeds to (almost always) silence.

    feeds  ->  blocking  ->  matcher agent  ->  triage agent  ->  alert or silence

Cost shape matters here. Blocking is free and runs over every recall in the
window. The matcher runs only on pairs that survived, and triage only on
confirmed matches. A sweep over a thousand recalls typically costs a handful of
model calls.
"""

from __future__ import annotations

from datetime import datetime

from .agents import Brain
from .blocking import generate_candidates
from .models import (
    Action,
    Alert,
    Hazard,
    MatchResult,
    RecallNotice,
    SilenceEntry,
    SweepReport,
    Verdict,
)
from .store import Store

# How many surviving pairs we are willing to spend model calls on in one sweep.
MAX_ADJUDICATIONS = 40


def run_sweep(
    store: Store,
    recalls: list[RecallNotice],
    brain: Brain,
    skip_seen: bool = True,
    max_adjudications: int = MAX_ADJUDICATIONS,
) -> SweepReport:
    """Read every recall, decide what (if anything) deserves a human."""
    started = datetime.now()
    inventory = store.inventory

    fresh = [r for r in recalls if not skip_seen or r.id not in store.seen_recalls]
    by_id = {r.id: r for r in fresh}

    candidates, silenced = generate_candidates(inventory, fresh)

    matches: list[MatchResult] = []
    adjudicated = 0
    for candidate in candidates[:max_adjudications]:
        item = store.item(candidate.item_id)
        recall = by_id.get(candidate.recall_id)
        if not item or not recall:
            continue

        judgement = brain.judge_match(item, recall, candidate)
        adjudicated += 1
        verdict = _coerce(Verdict, judgement.verdict, Verdict.NEED_INFO)

        if verdict is Verdict.NO_MATCH:
            silenced.append(
                SilenceEntry(
                    recall_id=recall.id,
                    recall_title=recall.title,
                    reason=judgement.reasoning,
                    stage="matcher",
                )
            )
            continue

        matches.append(
            MatchResult(
                item_id=item.id,
                recall_id=recall.id,
                verdict=verdict,
                confidence=judgement.confidence,
                reasoning=judgement.reasoning,
                missing_info=judgement.missing_info,
            )
        )

    alerts: list[Alert] = []
    for match in matches:
        item = store.item(match.item_id)
        recall = by_id.get(match.recall_id)
        if not item or not recall:
            continue

        if match.verdict is Verdict.NEED_INFO:
            # A question only the owner can answer is itself a real decision,
            # so it surfaces rather than being guessed at either way.
            alerts.append(
                Alert(
                    item_id=item.id,
                    recall_id=recall.id,
                    action=Action.NOTIFY_NOW,
                    hazard=_hazard_hint(recall),
                    headline=f"Check your {item.name} — it may be affected",
                    what_happened=match.reasoning,
                    what_to_do=match.missing_info
                    or "Check the model or serial number against the notice.",
                    confidence=match.confidence,
                    reasoning=match.reasoning,
                )
            )
            continue

        decision = brain.triage(item, recall, match)
        action = _coerce(Action, decision.action, Action.DIGEST)

        if action is Action.ARCHIVE:
            silenced.append(
                SilenceEntry(
                    recall_id=recall.id,
                    recall_title=recall.title,
                    reason=f"matched your {item.name}, but nothing for you to do: "
                    + decision.what_happened,
                    stage="triage",
                )
            )
            continue

        alerts.append(
            Alert(
                item_id=item.id,
                recall_id=recall.id,
                action=action,
                hazard=_coerce(Hazard, decision.hazard, Hazard.UNKNOWN),
                headline=decision.headline,
                what_happened=decision.what_happened,
                what_to_do=decision.what_to_do,
                draft_message=decision.draft_message,
                deadline=decision.deadline,
                confidence=match.confidence,
                reasoning=match.reasoning,
            )
        )

    new_alerts = store.add_alerts(alerts)
    store.add_silence(silenced)
    store.mark_seen([r.id for r in fresh])

    report = SweepReport(
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now().isoformat(timespec="seconds"),
        recalls_seen=len(fresh),
        candidates=len(candidates),
        adjudicated=adjudicated,
        alerts_raised=len(new_alerts),
        notify_now=sum(1 for a in new_alerts if a.action is Action.NOTIFY_NOW),
        silenced=len(silenced),
        sources=sorted({r.source.value for r in fresh}),
    )
    store.add_sweep(report)
    store.save()
    return report


def _coerce(enum_cls, value, default):
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        return default


def _hazard_hint(recall: RecallNotice) -> Hazard:
    """Rough severity from the notice text, used before triage has run."""
    text = (recall.hazard_text or recall.description or "").lower()
    if any(w in text for w in ("death", "fatal", "died", "suffocation", "strangulation")):
        return Hazard.DEATH
    if any(w in text for w in ("injury", "laceration", "burn", "fire", "fall", "crash")):
        return Hazard.INJURY
    if any(w in text for w in ("illness", "salmonella", "listeria", "contamination", "allergen")):
        return Hazard.ILLNESS
    return Hazard.UNKNOWN

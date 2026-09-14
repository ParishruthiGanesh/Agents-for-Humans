"""The cheap deterministic pass.

A sweep sees thousands of recalls. Sending every (item, recall) pair to a model
would be slow and expensive, so this module throws away the obvious misses
using rules only, and hands the survivors to the matcher agent.

Everything here is pure and testable: no model, no network.
"""

from __future__ import annotations

import re
from datetime import timedelta

from .models import Candidate, InventoryItem, RecallNotice, SilenceEntry

# Brands trade under several names on receipts and in recall notices alike.
BRAND_ALIASES: dict[str, str] = {
    "graco childrens products": "graco",
    "graco children's products": "graco",
    "fisher price": "fisher-price",
    "fisherprice": "fisher-price",
    "mattel fisher-price": "fisher-price",
    "philips respironics": "philips",
    "koninklijke philips": "philips",
    "abbott nutrition": "abbott",
    "abbott laboratories": "abbott",
    "peloton interactive": "peloton",
    "instant brands": "instant pot",
    "instant-pot": "instant pot",
    "britax child safety": "britax",
    "the boppy company": "boppy",
    "samsung electronics": "samsung",
    "best buy co": "best buy",
}

# Words that carry no discriminating signal in product names.
STOPWORDS = {
    "the", "and", "for", "with", "size", "new", "pack", "set", "of", "in",
    "by", "a", "an", "kit", "model", "series", "edition", "count", "ct",
    "oz", "lb", "inch", "black", "white", "grey", "gray", "blue", "red",
}

# How far outside a recall's stated sale window a purchase can fall before we
# treat the dates as disqualifying. Receipts and "sold from" dates rarely agree
# exactly, so this is deliberately generous.
DATE_SLACK = timedelta(days=180)

# Deliberately permissive. A pair wrongly dropped here is never seen again and
# nobody ever learns it was missed; a pair wrongly kept costs one model call.
# When in doubt, pass it up to the matcher.
MIN_SCORE = 0.25


def normalise_brand(value: str | None) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^a-z0-9 ']+", " ", value.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\b(inc|llc|ltd|corp|corporation|company|co|gmbh)\b", "", cleaned).strip()
    return BRAND_ALIASES.get(cleaned, cleaned)


def tokenise(value: str | None) -> set[str]:
    if not value:
        return set()
    words = re.findall(r"[a-z0-9][a-z0-9'-]*", value.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def model_numbers(value: str | None) -> set[str]:
    """Pull things that look like model or part numbers out of free text."""
    if not value:
        return set()
    found = set()
    for token in re.findall(r"\b[a-z]{0,3}[-]?\d{3,}[a-z0-9-]*\b", value.lower()):
        found.add(token.replace("-", ""))
    return found


def score_pair(item: InventoryItem, recall: RecallNotice) -> tuple[float, list[str]]:
    """Cheap similarity between something owned and something recalled.

    Returns a score in roughly 0..1 and the human-readable reasons behind it,
    which the silence log and the matcher prompt both reuse.
    """
    reasons: list[str] = []
    score = 0.0

    item_brand = normalise_brand(item.brand)
    recall_brands = {normalise_brand(b) for b in recall.brands} - {""}

    # Brand is the strongest cheap signal available.
    if item_brand and recall_brands:
        if item_brand in recall_brands:
            score += 0.55
            reasons.append(f"brand '{item.brand}' appears in the notice")
        elif any(item_brand in b or b in item_brand for b in recall_brands):
            score += 0.35
            reasons.append(f"brand '{item.brand}' partially matches the notice")
        else:
            # Different named brand is close to disqualifying, but recalls
            # sometimes name only the manufacturer, not the retail brand.
            reasons.append(f"brand '{item.brand}' not named in the notice")
    elif item_brand and item_brand in recall.searchable().lower():
        score += 0.4
        reasons.append(f"brand '{item.brand}' mentioned in the notice text")

    # Product-name overlap.
    item_tokens = tokenise(f"{item.name} {item.model or ''}")
    recall_tokens = tokenise(recall.searchable())
    if item_tokens:
        overlap = item_tokens & recall_tokens
        ratio = len(overlap) / len(item_tokens)
        if overlap:
            score += 0.35 * ratio
            reasons.append("shared product words: " + ", ".join(sorted(overlap)[:5]))

    # An exact model number is close to conclusive.
    item_models = model_numbers(f"{item.model or ''} {item.name}")
    recall_models = {m.lower().replace("-", "") for m in recall.models}
    recall_models |= model_numbers(recall.searchable())
    if item_models and (item_models & recall_models):
        score += 0.45
        reasons.append("model number appears in the notice")

    # Category agreement is a mild nudge, never decisive.
    if item.category and recall.category:
        if tokenise(item.category) & tokenise(recall.category):
            score += 0.1
            reasons.append("same product category")

    # Purchase date outside the stated sale window is disqualifying.
    sold_from, sold_to = recall.sold_window
    bought = item.purchase_date
    if bought and (sold_from or sold_to):
        too_early = sold_from and bought < sold_from - DATE_SLACK
        too_late = sold_to and bought > sold_to + DATE_SLACK
        if too_early or too_late:
            reasons.append(
                f"bought {item.purchased_on}, outside the affected window "
                f"{recall.sold_from or '?'} to {recall.sold_to or '?'}"
            )
            return 0.0, reasons
        reasons.append("purchase date falls inside the affected window")
        score += 0.1

    return min(score, 1.0), reasons


def generate_candidates(
    items: list[InventoryItem],
    recalls: list[RecallNotice],
    min_score: float = MIN_SCORE,
) -> tuple[list[Candidate], list[SilenceEntry]]:
    """Pair everything owned against everything recalled, keep the plausible.

    Returns the survivors plus a silence entry for each recall that no item
    came close to, so the person can see what was read and dismissed.
    """
    candidates: list[Candidate] = []
    silenced: list[SilenceEntry] = []
    live_items = [i for i in items if not i.disposed]

    for recall in recalls:
        best_score = 0.0
        best_reason = "nothing you own resembles this product"
        matched_any = False

        for item in live_items:
            score, reasons = score_pair(item, recall)
            if score >= min_score:
                matched_any = True
                candidates.append(
                    Candidate(
                        item_id=item.id,
                        recall_id=recall.id,
                        score=round(score, 3),
                        reasons=reasons,
                    )
                )
            elif score > best_score:
                best_score = score
                if reasons:
                    best_reason = reasons[-1]

        if not matched_any:
            silenced.append(
                SilenceEntry(
                    recall_id=recall.id,
                    recall_title=recall.title,
                    reason=best_reason,
                    stage="blocking",
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates, silenced

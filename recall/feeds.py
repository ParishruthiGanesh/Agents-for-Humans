"""Adapters over the public recall feeds.

Each adapter normalises a very different upstream shape into RecallNotice.
All three endpoints are free and need no API key:

  CPSC    https://www.saferproducts.gov/RestWebServices/Recall
  NHTSA   https://api.nhtsa.gov/recalls/recallsByVehicle
  openFDA https://api.fda.gov/{food,drug,device}/enforcement.json

Fetched pages are written to data/snapshots/ so a sweep can be replayed
offline and a demo never depends on someone else's uptime.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from .models import RecallNotice, Source

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"
USER_AGENT = "Recall/1.0 (hackathon project; public recall monitoring)"
TIMEOUT = 30


class FeedError(RuntimeError):
    pass


def _get_json(url: str, params: dict[str, str]) -> dict | list:
    query = urllib.parse.urlencode(params)
    full = f"{url}?{query}" if query else url
    request = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise FeedError(f"{url}: {exc}") from exc


def _first(value) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None


# --------------------------------------------------------------------------
# CPSC — consumer products
# --------------------------------------------------------------------------

def fetch_cpsc(since: date | None = None, limit: int = 200) -> list[RecallNotice]:
    since = since or (date.today() - timedelta(days=540))
    raw = _get_json(
        "https://www.saferproducts.gov/RestWebServices/Recall",
        {"format": "json", "RecallDateStart": since.isoformat()},
    )
    if not isinstance(raw, list):
        raise FeedError("CPSC returned an unexpected shape")
    return [_parse_cpsc(entry) for entry in raw[:limit]]


def _parse_cpsc(entry: dict) -> RecallNotice:
    products = entry.get("Products") or []
    hazards = entry.get("Hazards") or []
    remedies = entry.get("Remedies") or []
    manufacturers = entry.get("Manufacturers") or []

    brands, models, categories, units = [], [], [], None
    for product in products:
        name = product.get("Name")
        if name:
            brands.extend(_brand_guesses(name))
        if product.get("Model"):
            models.append(str(product["Model"]))
        if product.get("Type"):
            categories.append(product["Type"])
        units = units or product.get("NumberOfUnits")
    for maker in manufacturers:
        if maker.get("Name"):
            brands.append(maker["Name"])

    return RecallNotice(
        id=f"cpsc-{entry.get('RecallNumber') or entry.get('RecallID')}",
        source=Source.CPSC,
        title=entry.get("Title") or "Untitled CPSC recall",
        description=entry.get("Description") or "",
        published_on=(entry.get("RecallDate") or "")[:10] or None,
        brands=_dedupe(brands),
        models=_dedupe(models),
        category=categories[0] if categories else None,
        hazard_text="; ".join(h.get("Name", "") for h in hazards) or None,
        remedy_text="; ".join(r.get("Name", "") for r in remedies) or None,
        contact=entry.get("ConsumerContact"),
        url=entry.get("URL"),
        units=str(units) if units else None,
    )


def _brand_guesses(product_name: str) -> list[str]:
    """CPSC puts the brand inside the product name; the first words are it."""
    words = product_name.replace("®", " ").replace("™", " ").split()
    return [" ".join(words[:2])] if len(words) >= 2 else words


# --------------------------------------------------------------------------
# NHTSA — vehicles, tyres, child seats
# --------------------------------------------------------------------------

def fetch_nhtsa(make: str, model: str, year: int) -> list[RecallNotice]:
    raw = _get_json(
        "https://api.nhtsa.gov/recalls/recallsByVehicle",
        {"make": make, "model": model, "modelYear": str(year)},
    )
    results = raw.get("results", []) if isinstance(raw, dict) else []
    return [_parse_nhtsa(entry, make, model, year) for entry in results]


def _parse_nhtsa(entry: dict, make: str, model: str, year: int) -> RecallNotice:
    return RecallNotice(
        id=f"nhtsa-{entry.get('NHTSACampaignNumber')}",
        source=Source.NHTSA,
        title=entry.get("Summary", "")[:160] or f"{make} {model} recall",
        description=entry.get("Consequence") or entry.get("Summary") or "",
        published_on=(entry.get("ReportReceivedDate") or "")[:10] or None,
        brands=[make],
        models=[model],
        category=entry.get("Component") or "vehicle",
        hazard_text=entry.get("Consequence"),
        remedy_text=entry.get("Remedy"),
        contact=entry.get("Manufacturer"),
        url="https://www.nhtsa.gov/recalls",
    )


# --------------------------------------------------------------------------
# openFDA — food, drugs, devices
# --------------------------------------------------------------------------

_FDA_SOURCES = {
    "food": Source.FDA_FOOD,
    "drug": Source.FDA_DRUG,
    "device": Source.FDA_DEVICE,
}


def fetch_fda(endpoint: str = "food", limit: int = 100) -> list[RecallNotice]:
    if endpoint not in _FDA_SOURCES:
        raise FeedError(f"unknown openFDA endpoint {endpoint!r}")
    raw = _get_json(
        f"https://api.fda.gov/{endpoint}/enforcement.json",
        {"limit": str(limit)},
    )
    results = raw.get("results", []) if isinstance(raw, dict) else []
    return [_parse_fda(entry, endpoint) for entry in results]


def _parse_fda(entry: dict, endpoint: str) -> RecallNotice:
    brand = entry.get("recalling_firm") or ""
    return RecallNotice(
        id=f"fda-{endpoint}-{entry.get('recall_number')}",
        source=_FDA_SOURCES[endpoint],
        title=(entry.get("product_description") or "FDA enforcement report")[:160],
        description=entry.get("product_description") or "",
        published_on=_fda_date(entry.get("recall_initiation_date")),
        brands=_dedupe([brand]),
        models=_dedupe([entry.get("code_info", "")[:80]] if entry.get("code_info") else []),
        category=endpoint,
        hazard_text=entry.get("reason_for_recall"),
        remedy_text=f"Classification {entry.get('classification', 'unknown')}",
        contact=brand,
        url="https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts",
    )


def _fda_date(value: str | None) -> str | None:
    if value and len(value) == 8:
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------

def save_snapshot(name: str, notices: list[RecallNotice]) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    payload = [
        {**n.__dict__, "source": n.source.value} for n in notices
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_snapshots(directory: Path | None = None) -> list[RecallNotice]:
    directory = directory or SNAPSHOT_DIR
    notices: list[RecallNotice] = []
    for path in sorted(Path(directory).glob("*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8")):
            entry = dict(entry)
            entry["source"] = Source(entry["source"])
            notices.append(RecallNotice(**entry))
    return notices


def _dedupe(values: list[str]) -> list[str]:
    seen, out = set(), []
    for value in values:
        cleaned = (value or "").strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out

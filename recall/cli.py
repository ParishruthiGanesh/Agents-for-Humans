"""Command line entry point.

    recall seed                 load the demo household
    recall sweep [--offline]    read the feeds and decide what matters
    recall inbox                what needs you
    recall silence [n]          what was read and deliberately not shown
    recall add "<receipt>"      extract possessions from pasted text
    recall fetch                pull live recalls from CPSC / NHTSA / openFDA
    recall ask "<question>"     talk to the agent about your own things
    recall serve                run the web interface
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import build_brain
from .feeds import FeedError, fetch_cpsc, fetch_fda, fetch_nhtsa, load_snapshots, save_snapshot
from .models import Action, InventoryItem
from .pipeline import run_sweep
from .store import Store

ROOT = Path(__file__).resolve().parent.parent
DEMO_INVENTORY = ROOT / "data" / "demo_inventory.json"

DIM, BOLD, RED, YELLOW, GREEN, RESET = (
    "\033[2m", "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[0m"
)


def cmd_seed(args) -> int:
    store = Store()
    rows = json.loads(DEMO_INVENTORY.read_text(encoding="utf-8"))
    added = store.add_items([InventoryItem(**row) for row in rows])
    store.save()
    print(f"{GREEN}Loaded {len(added)} items{RESET} into {store.path}")
    for item in added:
        print(f"  {DIM}{item.id}{RESET}  {item.describe()}")
    return 0


def cmd_sweep(args) -> int:
    store = Store()
    if not store.inventory:
        print("Nothing to watch yet. Run `recall seed` or `recall add` first.")
        return 1

    recalls = load_snapshots()
    if not recalls:
        print("No recall corpus on disk. Run `recall fetch` first.")
        return 1

    brain = build_brain(offline=args.offline)
    mode = "replay" if args.offline else "Bedrock"
    print(f"{DIM}Sweeping {len(recalls)} recalls against {len(store.inventory)} "
          f"items ({mode})...{RESET}\n")

    report = run_sweep(store, recalls, brain, skip_seen=not args.rescan)

    print(f"{BOLD}Sweep complete{RESET}")
    print(f"  recalls read          {report.recalls_seen}")
    print(f"  survived cheap filter {report.candidates}")
    print(f"  sent to the matcher   {report.adjudicated}")
    print(f"  {BOLD}alerts raised         {report.alerts_raised}{RESET}"
          f"  ({report.notify_now} need you now)")
    print(f"  read and dismissed    {report.silenced}")
    if report.alerts_raised:
        print(f"\nRun {BOLD}recall inbox{RESET} to see them.")
    else:
        print(f"\n{GREEN}Nothing you own is affected.{RESET}")
    return 0


def cmd_inbox(args) -> int:
    store = Store()
    alerts = [a for a in store.alerts if not a.acknowledged]
    if not alerts:
        print(f"{GREEN}All clear.{RESET} "
              f"{store.total_recalls_seen} recalls read, none affect you.")
        return 0

    alerts.sort(key=lambda a: (a.action is not Action.NOTIFY_NOW, -a.hazard.rank))
    for alert in alerts:
        item = store.item(alert.item_id)
        colour = RED if alert.action is Action.NOTIFY_NOW else YELLOW
        tag = "NEEDS YOU NOW" if alert.action is Action.NOTIFY_NOW else "can wait"
        print(f"\n{colour}{BOLD}{alert.headline}{RESET}")
        print(f"{colour}[{tag}]{RESET} {DIM}{alert.hazard.value} hazard · "
              f"confidence {alert.confidence:.0%} · {alert.recall_id}{RESET}")
        print(f"  your item:  {item.describe() if item else alert.item_id}")
        print(f"  what:       {alert.what_happened}")
        print(f"  do this:    {alert.what_to_do}")
        if alert.deadline:
            print(f"  deadline:   {alert.deadline}")
        if alert.draft_message:
            print(f"\n  {DIM}draft message:{RESET}")
            for line in alert.draft_message.splitlines():
                print(f"    {DIM}{line}{RESET}")
    print()
    return 0


def cmd_silence(args) -> int:
    store = Store()
    entries = store.silence_log[-args.limit:]
    if not entries:
        print("Nothing read yet. Run `recall sweep` first.")
        return 0
    print(f"{BOLD}Read and deliberately not shown{RESET} "
          f"({len(store.silence_log)} total)\n")
    for entry in entries:
        print(f"  {entry.recall_title}")
        print(f"    {DIM}[{entry.stage}] {entry.reason}{RESET}")
    return 0


def cmd_add(args) -> int:
    store = Store()
    brain = build_brain(offline=args.offline)
    text = args.text
    if text == "-":
        text = sys.stdin.read()

    extracted = brain.extract_items(text)
    items = [
        InventoryItem(
            id=InventoryItem.make_id(e.brand, e.name, e.purchased_on),
            name=e.name, brand=e.brand, model=e.model, category=e.category,
            purchased_on=e.purchased_on, quantity=e.quantity,
            source_note="extracted from pasted text",
        )
        for e in extracted.items
    ]
    added = store.add_items(items)
    store.save()

    print(f"{GREEN}Added {len(added)} items{RESET}")
    for item in added:
        print(f"  {item.describe()}")
    if extracted.skipped:
        print(f"\n{DIM}Ignored:{RESET}")
        for line in extracted.skipped:
            print(f"  {DIM}{line}{RESET}")
    return 0


def cmd_fetch(args) -> int:
    total = 0
    try:
        cpsc = fetch_cpsc()
        save_snapshot("cpsc_live", cpsc)
        print(f"{GREEN}CPSC{RESET}    {len(cpsc)} notices")
        total += len(cpsc)
    except FeedError as exc:
        print(f"{RED}CPSC{RESET}    failed: {exc}")

    for endpoint in ("food", "drug", "device"):
        try:
            notices = fetch_fda(endpoint)
            save_snapshot(f"fda_{endpoint}_live", notices)
            print(f"{GREEN}openFDA{RESET} {len(notices)} {endpoint} notices")
            total += len(notices)
        except FeedError as exc:
            print(f"{RED}openFDA{RESET} {endpoint} failed: {exc}")

    store = Store()
    vehicles = [
        i for i in store.inventory
        if (i.category or "").lower() == "vehicle" and i.brand and i.model and i.purchased_on
    ]
    for vehicle in vehicles:
        year = int(vehicle.purchased_on[:4])
        try:
            notices = fetch_nhtsa(vehicle.brand, vehicle.model, year)
            save_snapshot(f"nhtsa_{vehicle.brand}_{vehicle.model}_{year}".lower(), notices)
            print(f"{GREEN}NHTSA{RESET}   {len(notices)} notices for "
                  f"{year} {vehicle.brand} {vehicle.model}")
            total += len(notices)
        except FeedError as exc:
            print(f"{RED}NHTSA{RESET}   {vehicle.brand} {vehicle.model} failed: {exc}")

    print(f"\n{total} notices on disk. Run `recall sweep` next.")
    return 0 if total else 1


def cmd_ask(args) -> int:
    from .tools import bind_store, build_assistant

    store = Store()
    bind_store(store)
    assistant = build_assistant()
    print(assistant(args.question))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    print(f"Recall is watching on {BOLD}http://127.0.0.1:{args.port}{RESET}")
    uvicorn.run("recall.api:app", host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recall", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="load the demo household").set_defaults(func=cmd_seed)

    sweep = sub.add_parser("sweep", help="read the feeds and decide what matters")
    sweep.add_argument("--offline", action="store_true", help="use recorded judgements")
    sweep.add_argument("--rescan", action="store_true", help="re-read recalls already seen")
    sweep.set_defaults(func=cmd_sweep)

    sub.add_parser("inbox", help="what needs you").set_defaults(func=cmd_inbox)

    silence = sub.add_parser("silence", help="what was read and not shown")
    silence.add_argument("limit", nargs="?", type=int, default=20)
    silence.set_defaults(func=cmd_silence)

    add = sub.add_parser("add", help="extract possessions from pasted text")
    add.add_argument("text", help="receipt or order email text, or - for stdin")
    add.add_argument("--offline", action="store_true")
    add.set_defaults(func=cmd_add)

    sub.add_parser("fetch", help="pull live recalls").set_defaults(func=cmd_fetch)

    ask = sub.add_parser("ask", help="talk to the agent about your things")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    serve = sub.add_parser("serve", help="run the web interface")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

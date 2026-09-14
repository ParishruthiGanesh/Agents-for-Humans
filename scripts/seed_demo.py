"""Build the offline demo corpus.

Writes three things, all with internally consistent ids:

  data/demo_inventory.json         a household's possessions
  data/snapshots/sample_corpus.json  recall notices to sweep against
  data/replay/*.json               recorded agent judgements for offline runs

The recall notices here are a CURATED SAMPLE modelled on the shape and language
of real public notices. Identifiers are illustrative, not official. Run
`recall fetch` to replace this corpus with live data from CPSC, NHTSA and
openFDA, which is what the agent reads in normal operation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recall.models import InventoryItem, to_jsonable  # noqa: E402

DATA = ROOT / "data"


def item(name, brand, model, category, purchased_on, **kw) -> InventoryItem:
    return InventoryItem(
        id=InventoryItem.make_id(brand, name, purchased_on),
        name=name,
        brand=brand,
        model=model,
        category=category,
        purchased_on=purchased_on,
        **kw,
    )


INVENTORY = [
    item("4Ever DLX 4-in-1 car seat", "Graco", "4Ever DLX", "car seat", "2020-03-14",
         source_note="order confirmation, March 2020"),
    item("BlendJet 2 portable blender", "BlendJet", "BlendJet 2", "blender", "2023-06-02",
         source_note="order confirmation, June 2023"),
    item("DreamStation CPAP machine", "Philips", "DreamStation", "medical device", "2019-11-20",
         serial="DS-4471992", source_note="added by hand"),
    item("Duo 7-in-1 pressure cooker 6qt", "Instant Pot", "Duo 60", "pressure cooker", "2021-01-08"),
    item("Accord Sport sedan", "Honda", "Accord", "vehicle", "2019-05-30",
         serial="1HGCV1F3XKA0••••", source_note="added by hand"),
    item("V8 Animal cordless vacuum", "Dyson", "V8 Animal", "vacuum", "2022-09-11"),
    item("Billy bookcase, white", "IKEA", "BILLY", "furniture", "2021-07-19"),
    item("WH-1000XM4 headphones", "Sony", "WH-1000XM4", "headphones", "2022-02-04"),
    item("Professional BN701 blender", "Ninja", "BN701", "blender", "2023-03-22"),
    item("Pura Vida espresso beans 1kg", "Lavazza", None, "coffee", "2024-04-02"),
    item("SoundLink Flex speaker", "Bose", "SoundLink Flex", "speaker", "2023-11-27"),
    item("Ridgeline cordless drill", "Ryobi", "P215K", "power tool", "2020-08-15"),
]

# Recall notices. Illustrative identifiers; realistic language and structure.
RECALLS = [
    {
        "id": "cpsc-24-2201",
        "source": "cpsc",
        "title": "BlendJet 2 portable blenders recalled due to fire and laceration hazards",
        "description": (
            "This recall involves BlendJet 2 portable blenders sold in a range of colours. "
            "The blender's battery can overheat, and the blades can fracture during use. "
            "Affected units were sold online and in retail stores nationwide."
        ),
        "published_on": "2024-03-14",
        "brands": ["BlendJet"],
        "models": ["BlendJet 2"],
        "category": "blender",
        "hazard_text": "Fire and laceration hazard; reports of burns and lacerations to users",
        "remedy_text": "Free replacement blender base from the manufacturer",
        "contact": "BlendJet customer support",
        "url": "https://www.cpsc.gov/Recalls",
        "units": "4,800,000",
        "sold_from": "2021-03-01",
        "sold_to": "2024-02-29",
    },
    {
        "id": "fda-device-z-1188-2021",
        "source": "fda_device",
        "title": "Philips Respironics DreamStation CPAP and BiPAP devices, sound abatement foam",
        "description": (
            "Certain Philips Respironics continuous and bi-level positive airway pressure "
            "devices contain polyester-based polyurethane sound abatement foam that may "
            "degrade into particles and release volatile organic compounds, which a user "
            "may inhale or swallow."
        ),
        "published_on": "2021-06-14",
        "brands": ["Philips Respironics", "Philips"],
        "models": ["DreamStation"],
        "category": "device",
        "hazard_text": "Possible inhalation of degraded foam particles and chemical emissions",
        "remedy_text": "Register the device for repair or replacement; consult your physician "
                       "before stopping prescribed therapy",
        "contact": "Philips Respironics",
        "url": "https://www.fda.gov/medical-devices",
        "sold_from": "2009-01-01",
        "sold_to": "2021-04-26",
    },
    {
        "id": "cpsc-19-2098",
        "source": "cpsc",
        "title": "Graco recalls certain 4Ever DLX convertible car seats over harness webbing",
        "description": (
            "The harness webbing on certain 4Ever DLX 4-in-1 car seats can loosen after "
            "adjustment, which may not restrain a child properly in a crash. Affected seats "
            "carry model numbers printed on a white label under the seat pad."
        ),
        "published_on": "2020-08-06",
        "brands": ["Graco", "Graco Children's Products"],
        "models": ["4Ever DLX", "2093054", "2093055"],
        "category": "car seat",
        "hazard_text": "Risk of injury in a crash if the child is not properly restrained",
        "remedy_text": "Free replacement harness kit",
        "contact": "Graco Children's Products",
        "url": "https://www.cpsc.gov/Recalls",
        "sold_from": "2019-01-01",
        "sold_to": "2021-03-31",
    },
    {
        "id": "nhtsa-19v-702",
        "source": "nhtsa",
        "title": "Honda Accord: software may cause the rear-view camera image to fail",
        "description": (
            "The rear-view camera display software may fail to show an image when the "
            "vehicle is in reverse, reducing the driver's view behind the vehicle."
        ),
        "published_on": "2019-09-25",
        "brands": ["Honda"],
        "models": ["Accord"],
        "category": "back over prevention: camera",
        "hazard_text": "Reduced rear visibility increases the risk of a crash",
        "remedy_text": "Dealers will update the rear-view camera software free of charge",
        "contact": "Honda customer service",
        "url": "https://www.nhtsa.gov/recalls",
        "sold_from": "2018-06-01",
        "sold_to": "2019-08-31",
    },
    {
        "id": "cpsc-23-1440",
        "source": "cpsc",
        "title": "Ryobi recalls cordless drills over switch defect",
        "description": (
            "The trigger switch on certain Ryobi 18V cordless drills can stick in the on "
            "position. Recalled drills carry model number P215Q printed on the housing."
        ),
        "published_on": "2023-05-11",
        "brands": ["Ryobi"],
        "models": ["P215Q"],
        "category": "power tool",
        "hazard_text": "Risk of injury from unexpected operation",
        "remedy_text": "Free repair kit",
        "contact": "Ryobi Tools",
        "url": "https://www.cpsc.gov/Recalls",
        "sold_from": "2022-01-01",
        "sold_to": "2023-04-30",
    },
    # --- notices that should never reach the person -----------------------
    {
        "id": "cpsc-24-1902",
        "source": "cpsc",
        "title": "Peloton recalls Tread+ treadmills after reports of injuries to children",
        "description": "Adult users, children and pets can be pulled under the rear of the treadmill.",
        "published_on": "2024-01-18", "brands": ["Peloton"], "models": ["Tread+"],
        "category": "exercise equipment",
        "hazard_text": "Risk of injury and death to children and pets",
        "remedy_text": "Full refund or free rear guard installation",
        "contact": "Peloton", "url": "https://www.cpsc.gov/Recalls",
    },
    {
        "id": "cpsc-23-0771",
        "source": "cpsc",
        "title": "Boppy recalls newborn loungers after infant deaths",
        "description": "Infants placed on the lounger can roll and suffocate against the padding.",
        "published_on": "2023-09-23", "brands": ["Boppy", "The Boppy Company"],
        "models": ["Newborn Lounger"], "category": "infant product",
        "hazard_text": "Suffocation hazard; infant fatalities reported",
        "remedy_text": "Stop use immediately and contact the firm for a refund",
        "contact": "The Boppy Company", "url": "https://www.cpsc.gov/Recalls",
    },
    {
        "id": "fda-food-f-0912-2024",
        "source": "fda_food",
        "title": "Fresh Express spinach, 10oz bags, potential Listeria monocytogenes",
        "description": "Bagged spinach distributed to retail, lot codes beginning with 24-",
        "published_on": "2024-02-09", "brands": ["Fresh Express"], "models": ["24-"],
        "category": "food", "hazard_text": "Possible Listeria monocytogenes contamination",
        "remedy_text": "Classification I", "contact": "Fresh Express",
        "url": "https://www.fda.gov/safety",
    },
    {
        "id": "nhtsa-22v-334",
        "source": "nhtsa",
        "title": "Ford F-150: engine block heater cable may corrode",
        "description": "The engine block heater cable can corrode and short circuit.",
        "published_on": "2022-05-17", "brands": ["Ford"], "models": ["F-150"],
        "category": "electrical system", "hazard_text": "Fire risk while plugged in",
        "remedy_text": "Dealers will inspect and replace the cable",
        "contact": "Ford", "url": "https://www.nhtsa.gov/recalls",
    },
    {
        "id": "cpsc-24-0455",
        "source": "cpsc",
        "title": "Baby Trend recalls high chairs over fall hazard",
        "description": "The high chair legs can detach during use.",
        "published_on": "2024-01-04", "brands": ["Baby Trend"], "models": ["Sit Right"],
        "category": "high chair", "hazard_text": "Fall hazard to children",
        "remedy_text": "Free repair kit", "contact": "Baby Trend",
        "url": "https://www.cpsc.gov/Recalls",
    },
    {
        "id": "fda-drug-d-2231-2024",
        "source": "fda_drug",
        "title": "Metformin hydrochloride extended-release tablets, NDMA above limit",
        "description": "Certain lots of metformin ER tablets, 500mg",
        "published_on": "2024-04-22", "brands": ["Marksans Pharma"], "models": ["500mg"],
        "category": "drug", "hazard_text": "NDMA content above the acceptable daily intake limit",
        "remedy_text": "Classification II", "contact": "Marksans Pharma",
        "url": "https://www.fda.gov/safety",
    },
    {
        "id": "cpsc-23-2277",
        "source": "cpsc",
        "title": "Onewheel self-balancing electric skateboards recalled after four deaths",
        "description": "The board can stop balancing while still moving, throwing the rider forward.",
        "published_on": "2023-09-29", "brands": ["Future Motion"], "models": ["Onewheel"],
        "category": "electric skateboard", "hazard_text": "Crash hazard; four deaths reported",
        "remedy_text": "Firmware update for some models, refund for others",
        "contact": "Future Motion", "url": "https://www.cpsc.gov/Recalls",
    },
    {
        "id": "fda-food-f-1204-2024",
        "source": "fda_food",
        "title": "Ground cinnamon, elevated lead levels, multiple distributors",
        "description": "Ground cinnamon products sold at discount retailers",
        "published_on": "2024-03-06", "brands": ["La Fiesta"], "models": [],
        "category": "food", "hazard_text": "Elevated lead levels",
        "remedy_text": "Classification II", "contact": "FDA",
        "url": "https://www.fda.gov/safety",
    },
    {
        "id": "cpsc-22-1683",
        "source": "cpsc",
        "title": "IKEA recalls MALM chests and dressers over tip-over hazard",
        "description": "Unsecured chests of drawers can tip over if not anchored to a wall.",
        "published_on": "2022-11-15", "brands": ["IKEA"], "models": ["MALM"],
        "category": "furniture", "hazard_text": "Tip-over and entrapment hazard to children",
        "remedy_text": "Free wall anchoring kit or refund", "contact": "IKEA",
        "url": "https://www.cpsc.gov/Recalls",
    },
]

# Recorded agent judgements, so the pipeline runs end to end with no credentials.
BLENDJET = InventoryItem.make_id("BlendJet", "BlendJet 2 portable blender", "2023-06-02")
CPAP = InventoryItem.make_id("Philips", "DreamStation CPAP machine", "2019-11-20")
CARSEAT = InventoryItem.make_id("Graco", "4Ever DLX 4-in-1 car seat", "2020-03-14")
ACCORD = InventoryItem.make_id("Honda", "Accord Sport sedan", "2019-05-30")
DRILL = InventoryItem.make_id("Ryobi", "Ridgeline cordless drill", "2020-08-15")
BILLY = InventoryItem.make_id("IKEA", "Billy bookcase, white", "2021-07-19")
NINJA = InventoryItem.make_id("Ninja", "Professional BN701 blender", "2023-03-22")

MATCHES = {
    f"{BLENDJET}|cpsc-24-2201": {
        "verdict": "match", "confidence": 0.96,
        "reasoning": "The notice names the BlendJet 2 by brand and model, and this unit was "
                     "bought in June 2023, inside the affected sale window.",
    },
    f"{CPAP}|fda-device-z-1188-2021": {
        "verdict": "match", "confidence": 0.93,
        "reasoning": "A Philips DreamStation bought in November 2019 falls squarely inside the "
                     "affected range for the sound abatement foam problem.",
    },
    f"{CARSEAT}|cpsc-19-2098": {
        "verdict": "need_info", "confidence": 0.62,
        "reasoning": "The recall covers only some 4Ever DLX seats, identified by model number. "
                     "The purchase date fits the window, but the model number on file is the "
                     "product family name rather than the specific number.",
        "missing_info": "Check the white label under the seat pad. If the model number is "
                        "2093054 or 2093055, this seat is affected.",
    },
    f"{ACCORD}|nhtsa-19v-702": {
        "verdict": "match", "confidence": 0.88,
        "reasoning": "A 2019 Honda Accord is covered by this rear-view camera software campaign.",
    },
    f"{BILLY}|cpsc-22-1683": {
        "verdict": "no_match", "confidence": 0.94,
        "reasoning": "Same brand and both are furniture, but the recall covers MALM chests of "
                     "drawers. A BILLY bookcase is a different product line and is not named "
                     "in the notice.",
    },
    f"{NINJA}|cpsc-24-2201": {
        "verdict": "no_match", "confidence": 0.97,
        "reasoning": "Both are blenders, but the recall is specific to BlendJet 2 units. A "
                     "Ninja BN701 is made by a different company and is not covered.",
    },
    f"{DRILL}|cpsc-23-1440": {
        "verdict": "no_match", "confidence": 0.91,
        "reasoning": "The recall covers model P215Q. This drill is a P215K, a different model, "
                     "and it was bought in 2020, before the affected units were sold.",
    },
}

TRIAGE = {
    f"{BLENDJET}|cpsc-24-2201": {
        "hazard": "injury", "action": "notify_now",
        "headline": "Your BlendJet 2 blender was recalled — stop using it",
        "what_happened": "The battery in this blender can overheat and the blades can break "
                         "apart while it is running. People have reported burns and cuts.",
        "what_to_do": "Stop using it today. BlendJet is giving affected owners a free "
                      "replacement base — request one through their recall page and keep your "
                      "order confirmation from June 2023 to hand.",
        "draft_message": "Hello,\n\nI own a BlendJet 2 portable blender, purchased on 2 June "
                         "2023. I understand it is covered by the recall over the battery and "
                         "blade defect. Please tell me how to claim the free replacement base.\n"
                         "\nMy order number is [ORDER NUMBER].\n\nThank you.",
    },
    f"{CPAP}|fda-device-z-1188-2021": {
        "hazard": "illness", "action": "notify_now",
        "headline": "Your Philips DreamStation is covered by the CPAP foam recall",
        "what_happened": "The sound-dampening foam inside these machines can break down over "
                         "time, and the particles and fumes can be breathed in. Philips is "
                         "repairing or replacing affected units.",
        "what_to_do": "Do not simply stop using it — this is prescribed therapy, so speak to "
                      "your doctor first. In the meantime, register the serial number "
                      "DS-4471992 on the Philips recall site to join the repair programme.",
        "draft_message": "Hello,\n\nI own a Philips DreamStation CPAP machine, serial "
                         "DS-4471992, purchased in November 2019. I would like to register it "
                         "for the sound abatement foam repair or replacement programme.\n\n"
                         "Please confirm my registration and the expected timescale.\n\nThank you.",
    },
    f"{ACCORD}|nhtsa-19v-702": {
        "hazard": "injury", "action": "digest",
        "headline": "Free camera software fix available for your 2019 Accord",
        "what_happened": "The rear-view camera can fail to show a picture when reversing. It is "
                         "a software fault, and dealers fix it free of charge.",
        "what_to_do": "No rush, but mention campaign 19V-702 next time the car is in for a "
                      "service. Any Honda dealer will do the update free.",
    },
}

EXTRACTIONS = {
    "blendjet": {
        "items": [
            {"name": "BlendJet 2 portable blender", "brand": "BlendJet", "model": "BlendJet 2",
             "category": "blender", "purchased_on": "2023-06-02", "quantity": 1},
            {"name": "WH-1000XM4 headphones", "brand": "Sony", "model": "WH-1000XM4",
             "category": "headphones", "purchased_on": "2023-06-02", "quantity": 1},
        ],
        "skipped": ["Shipping and handling — not a possession",
                    "Sales tax — not a possession"],
    }
}


def main() -> None:
    (DATA / "snapshots").mkdir(parents=True, exist_ok=True)
    (DATA / "replay").mkdir(parents=True, exist_ok=True)

    (DATA / "demo_inventory.json").write_text(
        json.dumps([to_jsonable(i) for i in INVENTORY], indent=2), encoding="utf-8"
    )
    (DATA / "snapshots" / "sample_corpus.json").write_text(
        json.dumps(RECALLS, indent=2), encoding="utf-8"
    )
    (DATA / "replay" / "matches.json").write_text(json.dumps(MATCHES, indent=2), encoding="utf-8")
    (DATA / "replay" / "triage.json").write_text(json.dumps(TRIAGE, indent=2), encoding="utf-8")
    (DATA / "replay" / "extractions.json").write_text(
        json.dumps(EXTRACTIONS, indent=2), encoding="utf-8"
    )
    print(f"seeded {len(INVENTORY)} items, {len(RECALLS)} recalls, "
          f"{len(MATCHES)} recorded match judgements")


if __name__ == "__main__":
    main()

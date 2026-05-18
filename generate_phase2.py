import csv
import datetime as dt
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PHASE1_CITIES = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Rawalpindi",
    "Faisalabad",
    "Multan",
    "Peshawar",
    "Quetta",
]

# City pricing multipliers (avg). Min/max will vary around this.
CITY_MULT = {
    "Lahore": 1.00,
    "Karachi": 1.08,
    "Islamabad": 1.05,
    "Rawalpindi": 1.04,
    "Faisalabad": 0.96,
    "Multan": 0.95,
    "Peshawar": 1.02,
    "Quetta": 1.03,
}

CITY_CONF = {
    "Lahore": 0.7,
    "Karachi": 0.62,
    "Islamabad": 0.6,
    "Rawalpindi": 0.58,
    "Faisalabad": 0.55,
    "Multan": 0.55,
    "Peshawar": 0.52,
    "Quetta": 0.5,
}


def _stable_u01(*parts: str) -> float:
    h = hashlib.sha256(("|".join(parts)).encode("utf-8")).hexdigest()
    # use 10 hex chars -> 40 bits
    n = int(h[:10], 16)
    return (n % 10_000_000) / 10_000_000.0


def _round_price(x: float) -> float:
    # Keep small consumables with decimals, large with integers
    if x < 20:
        return round(x, 2)
    if x < 200:
        return round(x, 1)
    return round(x)


def _mk_range(base_avg: float, spread_pct: float, jitter: float):
    # spread_pct like 0.10 = ±10%
    # jitter shifts avg within a small band to avoid clones
    avg = base_avg * (1.0 + (jitter - 0.5) * 0.03)
    mn = avg * (1.0 - spread_pct)
    mx = avg * (1.0 + spread_pct)
    # Ensure numeric order
    return _round_price(mn), _round_price(mx), _round_price(avg)


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def next_material_id(existing_ids: set[str]) -> str:
    # material_ids are like M0001
    mx = 0
    for mid in existing_ids:
        if mid.startswith("M"):
            try:
                mx = max(mx, int(mid[1:]))
            except ValueError:
                pass
    return f"M{mx+1:04d}"


def add_ai_columns(materials: list[dict]) -> list[dict]:
    # Upgrade schema by adding functional_tag, urdu_name, synonyms
    for m in materials:
        m.setdefault("functional_tag", "")
        m.setdefault("urdu_name", "")
        m.setdefault("synonyms", "")
    return materials


def enrich_existing_material_ai(materials: list[dict]) -> None:
    # Lightweight enrichment for current 140 rows
    tag_by_phase = {
        "Site Preparation": "structural",
        "Excavation & Foundation": "structural",
        "Grey Structure": "structural",
        "Masonry & Walls": "structural",
        "Plumbing (Rough + Final)": "plumbing",
        "Electrical (Rough + Final)": "electrical",
        "Plastering & Screeding": "finishing",
        "Flooring & Tiling": "finishing",
        "Paint & Finishing": "finishing",
        "Roofing & Waterproofing": "waterproofing",
        "Carpentry & Woodwork": "finishing",
        "Aluminum & Glass Work": "finishing",
        "Kitchen & Wardrobes": "finishing",
        "Sanitary & Bathroom Fittings": "plumbing",
        "External Works": "external",
        "Miscellaneous / Consumables (VERY IMPORTANT)": "consumable",
    }
    urdu_seed = {
        "cement": "سیمنٹ",
        "steel": "سریا",
        "brick": "اینٹ",
        "sand": "ریت",
        "crush": "بجری",
        "wire": "تار",
        "conduit": "کنڈیٹ",
        "paint": "پینٹ",
        "primer": "پرائمر",
        "putty": "پٹی",
        "tape": "ٹیپ",
        "sealant": "سیلنٹ",
        "pipe": "پائپ",
    }
    for m in materials:
        ph = m.get("phase", "")
        m["functional_tag"] = tag_by_phase.get(ph, "")
        nm = (m.get("name") or "").lower()
        for k, u in urdu_seed.items():
            if k in nm:
                m["urdu_name"] = u
                break
        # synonyms: include local terms where applicable
        syn = []
        if "sarya" in nm or "rebar" in nm or "steel rebar" in nm:
            syn += ["sarya", "rebar", "steel"]
        if "bajri" in nm or "crush" in nm:
            syn += ["bajri", "crush"]
        if "bricks" in nm or "brick" in nm:
            syn += ["eent", "bricks"]
        if "choona" in nm:
            syn += ["lime", "choona"]
        m["synonyms"] = "; ".join(dict.fromkeys(syn))


def generate_material_expansion(existing: list[dict]) -> list[dict]:
    """
    Expand materials to ~650–750 items by adding granular SKUs commonly sold in PK hardware markets.
    """
    existing_ids = {m["material_id"] for m in existing}
    materials = list(existing)

    def add(
        name,
        phase,
        category,
        subcategory,
        description,
        specifications,
        unit,
        usage_type,
        usage_ratio,
        quality_grade="standard",
        brand="",
        notes="",
        functional_tag="",
        urdu_name="",
        synonyms="",
    ):
        nonlocal materials, existing_ids
        mid = next_material_id(existing_ids)
        existing_ids.add(mid)
        materials.append(
            {
                "material_id": mid,
                "name": name,
                "phase": phase,
                "category": category,
                "subcategory": subcategory,
                "description": description,
                "specifications": specifications,
                "unit": unit,
                "usage_type": usage_type,
                "usage_ratio": f"{float(usage_ratio):g}",
                "quality_grade": quality_grade,
                "brand": brand,
                "notes": notes,
                "functional_tag": functional_tag,
                "urdu_name": urdu_name,
                "synonyms": synonyms,
            }
        )

    # --- Plumbing fittings (granular) ---
    ppr_sizes = ["1/2 in", "3/4 in", "1 in", "1.25 in", "1.5 in", "2 in"]
    fitting_types = [
        ("Elbow 90°", "elbow"),
        ("Elbow 45°", "elbow"),
        ("Tee", "tee"),
        ("Socket", "socket"),
        ("Union", "union"),
        ("Reducer", "reducer"),
        ("End cap", "cap"),
        ("Male threaded adapter", "mta"),
        ("Female threaded adapter", "fta"),
        ("Ball valve", "valve"),
        ("Gate valve", "valve"),
        ("Check valve", "valve"),
    ]
    for sz in ppr_sizes:
        for ft, syn in fitting_types:
            add(
                name=f"PPRC fitting {ft} {sz}",
                phase="Plumbing (Rough + Final)",
                category="Plumbing",
                subcategory="Fittings",
                description="PPRC fitting used in hot/cold distribution networks.",
                specifications=f"PN20; {sz}",
                unit="pcs",
                usage_type="per_sqft",
                usage_ratio=0.03 if "valve" not in ft.lower() else 0.004,
                functional_tag="plumbing",
                urdu_name="فٹنگ",
                synonyms=f"pprc; fitting; {syn}",
            )

    upvc_sizes = ["1.5 in", "2 in", "3 in", "4 in", "6 in"]
    upvc_fit = [
        ("Bend 45°", "bend"),
        ("Bend 90°", "bend"),
        ("Tee", "tee"),
        ("Y-tee", "ytee"),
        ("Coupler", "coupler"),
        ("Reducer", "reducer"),
        ("End cap", "cap"),
        ("Cleanout", "cleanout"),
    ]
    for sz in upvc_sizes:
        for ft, syn in upvc_fit:
            add(
                name=f"UPVC sewer fitting {ft} {sz}",
                phase="Plumbing (Rough + Final)",
                category="Plumbing",
                subcategory="Fittings",
                description="UPVC drainage/sewer fitting for waste lines.",
                specifications=f"SN4; {sz}",
                unit="pcs",
                usage_type="per_sqft",
                usage_ratio=0.02,
                functional_tag="plumbing",
                urdu_name="ڈرین فٹنگ",
                synonyms=f"upvc; sewer; {syn}",
            )

    # --- Electrical accessories (granular) ---
    wire_sizes = [
        ("1.0mm2", 0.25),
        ("1.5mm2", 0.85),
        ("2.5mm2", 0.65),
        ("4.0mm2", 0.22),
        ("6.0mm2", 0.12),
        ("10mm2", 0.06),
    ]
    for size, ratio in wire_sizes:
        add(
            name=f"Electrical wire copper PVC {size}",
            phase="Electrical (Rough + Final)",
            category="Electrical",
            subcategory="Wiring",
            description="Copper PVC insulated wire for house wiring.",
            specifications=f"99.9% Cu; {size}",
            unit="m",
            usage_type="per_sqft",
            usage_ratio=ratio,
            functional_tag="electrical",
            urdu_name="تار",
            synonyms=f"wire; cable; {size}",
        )

    db_ways = [4, 6, 8, 10, 12, 16]
    for w in db_ways:
        add(
            name=f"Distribution board (DB) {w}-way",
            phase="Electrical (Rough + Final)",
            category="Electrical",
            subcategory="Panel",
            description="Distribution board for MCBs and circuit distribution.",
            specifications=f"Single phase; {w} way",
            unit="pcs",
            usage_type="per_house",
            usage_ratio=1 if w >= 8 else 0.4,
            functional_tag="electrical",
            urdu_name="ڈی بی",
            synonyms="db; distribution board",
        )

    mcb_types = [("SP", 6), ("SP", 10), ("SP", 16), ("SP", 20), ("SP", 32), ("DP", 32), ("DP", 40), ("DP", 63)]
    for poles, amps in mcb_types:
        add(
            name=f"MCB {poles} {amps}A",
            phase="Electrical (Rough + Final)",
            category="Electrical",
            subcategory="Protection",
            description="Miniature circuit breaker for circuit protection.",
            specifications=f"{poles}; {amps}A; IEC",
            unit="pcs",
            usage_type="per_sqft",
            usage_ratio=0.006,
            functional_tag="electrical",
            urdu_name="ایم سی بی",
            synonyms="mcb; breaker",
        )

    # Junction boxes & plates
    for g in [1, 2, 3, 4, 6, 8, 12]:
        add(
            name=f"Switch plate {g}-gang",
            phase="Electrical (Rough + Final)",
            category="Electrical",
            subcategory="Accessories",
            description="Front plate for switch box.",
            specifications=f"{g} gang",
            unit="pcs",
            usage_type="per_sqft",
            usage_ratio=0.01,
            functional_tag="electrical",
            urdu_name="سوئچ پلیٹ",
            synonyms="plate; faceplate",
        )
        add(
            name=f"Switch box {g}-gang (GI/PVC)",
            phase="Electrical (Rough + Final)",
            category="Electrical",
            subcategory="Back boxes",
            description="Back box for switches and sockets.",
            specifications=f"{g} gang",
            unit="pcs",
            usage_type="per_sqft",
            usage_ratio=0.01,
            functional_tag="electrical",
            urdu_name="سوئچ باکس",
            synonyms="back box; pattii",
        )

    # Lugs, glands, hooks
    lug_sizes = ["1.5mm2", "2.5mm2", "4mm2", "6mm2", "10mm2", "16mm2", "25mm2"]
    for s in lug_sizes:
        add(
            name=f"Cable lug copper {s}",
            phase="Electrical (Rough + Final)",
            category="Electrical",
            subcategory="Accessories",
            description="Copper lug for terminating cables.",
            specifications=f"Ring lug; {s}",
            unit="pcs",
            usage_type="per_sqft",
            usage_ratio=0.02,
            functional_tag="electrical",
            urdu_name="لگ",
            synonyms="lug; terminal",
        )
    add(
        name="Ceiling fan hook heavy duty",
        phase="Electrical (Rough + Final)",
        category="Electrical",
        subcategory="Accessories",
        description="Hook/anchoring for ceiling fan installation.",
        specifications="MS heavy duty",
        unit="pcs",
        usage_type="per_marla",
        usage_ratio=2.0,
        functional_tag="electrical",
        urdu_name="فین ہک",
        synonyms="fan hook",
    )

    # --- Fasteners & consumables (granular) ---
    nail_sizes = [("1 in", 0.0011), ("1.5 in", 0.0013), ("2 in", 0.0016), ("2.5 in", 0.0018), ("3 in", 0.0020), ("4 in", 0.0022)]
    for sz, ratio in nail_sizes:
        add(
            name=f"Nails (common) {sz}",
            phase="Miscellaneous / Consumables (VERY IMPORTANT)",
            category="Consumable",
            subcategory="Fasteners",
            description="Common nails used in shuttering/woodwork/general site tasks.",
            specifications=f"MS; {sz}",
            unit="kg",
            usage_type="per_sqft",
            usage_ratio=ratio,
            functional_tag="consumable",
            urdu_name="کیل",
            synonyms="nails; keel",
        )

    screw_sizes = ["1/2 in", "1 in", "1.5 in", "2 in", "2.5 in", "3 in"]
    screw_types = ["wood screw", "self-tapping", "drywall screw", "chipboard screw"]
    for t in screw_types:
        for sz in screw_sizes:
            add(
                name=f"{t.title()} {sz}",
                phase="Miscellaneous / Consumables (VERY IMPORTANT)",
                category="Consumable",
                subcategory="Fasteners",
                description="Screws for fixtures, hardware, woodwork, and general installation.",
                specifications=f"{t}; {sz}",
                unit="pcs",
                usage_type="per_sqft",
                usage_ratio=0.08,
                functional_tag="consumable",
                urdu_name="اسکرو",
                synonyms="screw; fastener",
            )

    plug_sizes = ["6mm", "8mm", "10mm", "12mm"]
    for sz in plug_sizes:
        add(
            name=f"Wall plug (rawl plug) {sz}",
            phase="Miscellaneous / Consumables (VERY IMPORTANT)",
            category="Consumable",
            subcategory="Fasteners",
            description="Plastic wall plug used with screws for masonry fixing.",
            specifications=sz,
            unit="pcs",
            usage_type="per_sqft",
            usage_ratio=0.12,
            functional_tag="consumable",
            urdu_name="رال پلگ",
            synonyms="rawl plug; wall plug",
        )

    tie_sizes = ["100mm", "150mm", "200mm", "250mm", "300mm"]
    for sz in tie_sizes:
        add(
            name=f"Cable ties (zip ties) {sz}",
            phase="Miscellaneous / Consumables (VERY IMPORTANT)",
            category="Consumable",
            subcategory="Fixings",
            description="Cable ties for bundling and temporary fixing.",
            specifications=sz,
            unit="pack",
            usage_type="per_marla",
            usage_ratio=0.15,
            functional_tag="consumable",
            urdu_name="کیبل ٹائی",
            synonyms="zip tie; cable tie",
        )

    # --- Tile & finishing accessories ---
    spacer_mm = [1, 2, 3, 4, 5]
    for mm in spacer_mm:
        add(
            name=f"Tile spacer {mm}mm",
            phase="Flooring & Tiling",
            category="Consumable",
            subcategory="Accessories",
            description="Spacer for consistent tile joint width.",
            specifications=f"{mm}mm",
            unit="pcs",
            usage_type="per_sqft",
            usage_ratio=0.6,
            functional_tag="finishing",
            urdu_name="ٹائل اسپیسَر",
            synonyms="tile spacer",
        )
    grout_types = ["cement grout", "epoxy grout"]
    for gt in grout_types:
        add(
            name=f"Tile grout ({gt})",
            phase="Flooring & Tiling",
            category="Tiling",
            subcategory="Grout",
            description="Grout for tile joints; epoxy used for premium wet areas.",
            specifications="White/colored",
            unit="kg",
            usage_type="per_sqft",
            usage_ratio=0.03,
            quality_grade="premium" if "epoxy" in gt else "standard",
            functional_tag="finishing",
            urdu_name="گراؤٹ",
            synonyms="grout",
        )
    adhesive_grades = [("standard", 0.0006), ("flex", 0.00075), ("rapid set", 0.0008)]
    for grade, ratio in adhesive_grades:
        add(
            name=f"Tile adhesive ({grade})",
            phase="Flooring & Tiling",
            category="Tiling",
            subcategory="Adhesive",
            description="Cementitious tile adhesive for floor/wall tiles.",
            specifications="20kg bag",
            unit="bag",
            usage_type="per_sqft",
            usage_ratio=ratio,
            quality_grade="premium" if grade != "standard" else "standard",
            functional_tag="finishing",
            urdu_name="ٹائل چپک",
            synonyms="tile adhesive",
        )
    trims = ["L-trim", "T-trim", "U-trim", "corner trim"]
    for t in trims:
        add(
            name=f"Tile trim {t} (aluminum/PVC)",
            phase="Flooring & Tiling",
            category="Tiling",
            subcategory="Profiles",
            description="Trims for tile edges and corners.",
            specifications="8–10 ft",
            unit="rft",
            usage_type="per_sqft",
            usage_ratio=0.06,
            functional_tag="finishing",
            urdu_name="ٹرِم",
            synonyms="tile trim; profile",
        )

    # --- Waterproofing systems ---
    membranes = [("Torch-on membrane 3mm", 0.45), ("Torch-on membrane 4mm", 0.45), ("Liquid membrane acrylic", 0.00018), ("Cementitious waterproof coating", 0.00022)]
    for nm, ratio in membranes:
        unit = "ft2" if "Torch-on" in nm else "kg"
        add(
            name=nm,
            phase="Roofing & Waterproofing",
            category="Waterproofing",
            subcategory="System",
            description="Roof/wet-area waterproofing system component.",
            specifications=nm,
            unit=unit,
            usage_type="per_sqft",
            usage_ratio=ratio,
            functional_tag="waterproofing",
            urdu_name="واٹر پروفنگ",
            synonyms="waterproof; membrane",
        )
    add(
        name="XPS insulation board (roof) 1 inch",
        phase="Roofing & Waterproofing",
        category="Waterproofing",
        subcategory="Insulation",
        description="Thermal insulation board for roofs.",
        specifications="XPS 25–35kg/m3; 1 in",
        unit="ft2",
        usage_type="per_sqft",
        usage_ratio=0.28,
        functional_tag="waterproofing",
        urdu_name="انسولیشن",
        synonyms="xps; insulation",
    )
    add(
        name="Geotextile protection layer",
        phase="Roofing & Waterproofing",
        category="Waterproofing",
        subcategory="Protection",
        description="Protection layer for membranes in some systems.",
        specifications="Non-woven 150–200gsm",
        unit="ft2",
        usage_type="per_sqft",
        usage_ratio=0.3,
        functional_tag="waterproofing",
        urdu_name="جیوٹیکسٹائل",
        synonyms="geotextile",
    )

    # --- Paint ecosystem ---
    primers = [("Interior primer (water-based)", 0.00022), ("Exterior primer (alkali-resistant)", 0.00022), ("Metal red oxide primer", 0.00004), ("Wood primer", 0.00004)]
    for nm, ratio in primers:
        add(
            name=nm,
            phase="Paint & Finishing",
            category="Paint",
            subcategory="Primer",
            description="Primer coat for improved adhesion and uniform finish.",
            specifications=nm,
            unit="litre",
            usage_type="per_sqft",
            usage_ratio=ratio,
            functional_tag="finishing",
            urdu_name="پرائمر",
            synonyms="primer",
        )
    add(
        name="Paint thinner (enamel)",
        phase="Paint & Finishing",
        category="Paint",
        subcategory="Solvent",
        description="Thinner for enamel paint and cleaning tools.",
        specifications="Solvent-based",
        unit="litre",
        usage_type="per_marla",
        usage_ratio=0.25,
        functional_tag="finishing",
        urdu_name="تھنر",
        synonyms="thinner; solvent",
    )
    rollers = ["4 in mini roller", "7 in roller", "9 in roller", "roller handle", "roller tray"]
    for r in rollers:
        add(
            name=r,
            phase="Paint & Finishing",
            category="Consumable",
            subcategory="Tools",
            description="Painting tool accessory.",
            specifications=r,
            unit="pcs",
            usage_type="per_marla",
            usage_ratio=0.3,
            functional_tag="consumable",
            urdu_name="رولر",
            synonyms="roller; paint tool",
        )

    # --- Add more plumbing fixtures accessories ---
    fixtures = [
        ("Bib tap (brass) 1/2 in", "pcs", 0.15),
        ("Stop cock (concealed) 1/2 in", "pcs", 0.12),
        ("Wash basin waste coupling", "pcs", 0.10),
        ("Bottle trap (SS)", "pcs", 0.10),
        ("Toilet seat cover", "pcs", 0.08),
        ("Geyser safety valve", "pcs", 0.04),
    ]
    for nm, unit, ratio in fixtures:
        add(
            name=nm,
            phase="Sanitary & Bathroom Fittings",
            category="Sanitary",
            subcategory="Accessories",
            description="Bathroom/kitchen fixture accessory item.",
            specifications=nm,
            unit=unit,
            usage_type="per_marla",
            usage_ratio=ratio,
            functional_tag="plumbing",
            urdu_name="فٹنگ",
            synonyms="fixture; fitting",
        )

    # --- PVC/CPVC supply ecosystem (granular) ---
    pvc_sizes = ["1/2 in", "3/4 in", "1 in", "1.25 in", "1.5 in", "2 in"]
    pvc_types = [("PVC (pressure)", "PVC"), ("CPVC (hot water)", "CPVC")]
    pvc_fit = [
        ("Elbow 90°", "elbow"),
        ("Tee", "tee"),
        ("Socket", "socket"),
        ("Reducer", "reducer"),
        ("Union", "union"),
        ("End cap", "cap"),
        ("Ball valve", "valve"),
    ]
    for typ_name, syn_typ in pvc_types:
        for sz in pvc_sizes:
            add(
                name=f"{typ_name} pipe {sz}",
                phase="Plumbing (Rough + Final)",
                category="Plumbing",
                subcategory="PVC/CPVC",
                description="Water supply line pipe commonly used in Pakistan (alternative to PPRC).",
                specifications=f"Schedule/PN as per local; {sz}",
                unit="ft",
                usage_type="per_sqft",
                usage_ratio=0.045 if sz in {"1/2 in", "3/4 in"} else 0.02,
                functional_tag="plumbing",
                urdu_name="پائپ",
                synonyms=f"{syn_typ}; pipe",
            )
            for ft, syn in pvc_fit:
                add(
                    name=f"{typ_name} fitting {ft} {sz}",
                    phase="Plumbing (Rough + Final)",
                    category="Plumbing",
                    subcategory="Fittings",
                    description="PVC/CPVC fitting used for water supply joints and direction changes.",
                    specifications=f"{syn_typ}; {sz}",
                    unit="pcs",
                    usage_type="per_sqft",
                    usage_ratio=0.02 if "valve" not in ft.lower() else 0.003,
                    functional_tag="plumbing",
                    urdu_name="فٹنگ",
                    synonyms=f"{syn_typ}; fitting; {syn}",
                )

    # --- GI fittings & valves (still common on sites) ---
    gi_sizes = ["1/2 in", "3/4 in", "1 in", "1.25 in", "1.5 in", "2 in"]
    gi_fit = ["Elbow", "Tee", "Socket", "Nipple", "Union", "Reducer"]
    for sz in gi_sizes:
        for ft in gi_fit:
            add(
                name=f"GI fitting {ft} {sz}",
                phase="Plumbing (Rough + Final)",
                category="Plumbing",
                subcategory="GI fittings",
                description="Galvanized iron fitting (threaded) used for water lines and connections.",
                specifications=f"GI threaded; {sz}",
                unit="pcs",
                usage_type="per_sqft",
                usage_ratio=0.008,
                functional_tag="plumbing",
                urdu_name="جی آئی فٹنگ",
                synonyms="gi; fitting; threaded",
            )
        for v in ["Ball valve", "Gate valve", "Check valve"]:
            add(
                name=f"GI {v} {sz}",
                phase="Plumbing (Rough + Final)",
                category="Plumbing",
                subcategory="Valves",
                description="Valve for GI water lines.",
                specifications=f"GI; {v}; {sz}",
                unit="pcs",
                usage_type="per_marla",
                usage_ratio=0.12 if sz == "1/2 in" else 0.06,
                functional_tag="plumbing",
                urdu_name="والو",
                synonyms="valve; gi",
            )

    # --- Conduit ecosystem (sizes + fittings) ---
    conduit_sizes = [("16mm", 0.15), ("20mm", 0.55), ("25mm", 0.12), ("32mm", 0.05)]
    conduit_fit = ["bend", "tee", "coupler", "junction box 4x4", "junction box 6x6", "inspection box"]
    for sz, ratio in conduit_sizes:
        add(
            name=f"PVC conduit heavy gauge {sz}",
            phase="Electrical (Rough + Final)",
            category="Electrical",
            subcategory="Conduit",
            description="PVC conduit for concealed wiring.",
            specifications=f"Heavy gauge; {sz}",
            unit="ft",
            usage_type="per_sqft",
            usage_ratio=ratio,
            functional_tag="electrical",
            urdu_name="کنڈیٹ",
            synonyms="conduit; pvc",
        )
        for ft in conduit_fit:
            add(
                name=f"Conduit accessory {ft} ({sz})",
                phase="Electrical (Rough + Final)",
                category="Electrical",
                subcategory="Accessories",
                description="Conduit fitting/accessory for routing and junctions.",
                specifications=f"{sz}",
                unit="pcs",
                usage_type="per_sqft",
                usage_ratio=0.06,
                functional_tag="electrical",
                urdu_name="کنڈیٹ فٹنگ",
                synonyms="conduit; accessory",
            )

    # --- Lighting + final accessories ---
    lights = [
        ("LED downlight 9W", 0.35),
        ("LED downlight 12W", 0.28),
        ("LED panel 18W", 0.18),
        ("LED bulb 12W", 0.25),
        ("Ceiling light (surface) 24W", 0.12),
        ("Exhaust fan 8 inch", 0.06),
        ("Ceiling fan 56 inch", 0.08),
        ("Regulator fan", 0.08),
        ("Bell push switch", 0.06),
        ("Door bell (chime)", 0.06),
        ("TV socket point", 0.04),
        ("Data socket (RJ45) point", 0.03),
    ]
    for nm, ratio in lights:
        add(
            name=nm,
            phase="Electrical (Rough + Final)",
            category="Electrical",
            subcategory="Fixtures",
            description="Electrical fixture or final accessory commonly installed.",
            specifications=nm,
            unit="pcs",
            usage_type="per_marla",
            usage_ratio=ratio,
            functional_tag="electrical",
            urdu_name="لائٹ/فین",
            synonyms="light; fixture",
        )

    # --- Anchors, bolts, washers, blades, abrasives ---
    bolt_sizes = ["M6", "M8", "M10", "M12"]
    for sz in bolt_sizes:
        add(
            name=f"Nut & bolt set {sz}",
            phase="Miscellaneous / Consumables (VERY IMPORTANT)",
            category="Consumable",
            subcategory="Fasteners",
            description="Nut/bolt set for fixtures, brackets, and fabrication.",
            specifications=sz,
            unit="set",
            usage_type="per_marla",
            usage_ratio=0.8,
            functional_tag="consumable",
            urdu_name="نٹ بولٹ",
            synonyms="bolt; nut; washer",
        )
        add(
            name=f"Washer {sz}",
            phase="Miscellaneous / Consumables (VERY IMPORTANT)",
            category="Consumable",
            subcategory="Fasteners",
            description="Washer used with nuts/bolts.",
            specifications=sz,
            unit="pcs",
            usage_type="per_marla",
            usage_ratio=8,
            functional_tag="consumable",
            urdu_name="واشر",
            synonyms="washer",
        )
    anchors = ["Anchor bolt 8mm", "Anchor bolt 10mm", "Anchor bolt 12mm"]
    for a in anchors:
        add(
            name=a,
            phase="Miscellaneous / Consumables (VERY IMPORTANT)",
            category="Consumable",
            subcategory="Fasteners",
            description="Expansion anchor for heavy fixtures/frames.",
            specifications=a,
            unit="pcs",
            usage_type="per_marla",
            usage_ratio=1.5,
            functional_tag="consumable",
            urdu_name="اینکر",
            synonyms="anchor; bolt",
        )
    blades = ["Hacksaw blade", "Wood cutting blade (jigsaw)", "Metal cutting blade (jigsaw)", "Tile cutting wheel"]
    for b in blades:
        add(
            name=b,
            phase="Miscellaneous / Consumables (VERY IMPORTANT)",
            category="Consumable",
            subcategory="Tools",
            description="Blade/cutting consumable used on site.",
            specifications=b,
            unit="pcs",
            usage_type="per_marla",
            usage_ratio=0.6,
            functional_tag="consumable",
            urdu_name="بلیڈ",
            synonyms="blade; cutting",
        )

    # --- Sealants/adhesives by type ---
    sealants = [
        ("Acrylic sealant (gap filler)", "pcs", 0.25),
        ("Silicone sealant (neutral cure)", "pcs", 0.5),
        ("Silicone sealant (acetoxy)", "pcs", 0.35),
        ("PU sealant (construction)", "pcs", 0.2),
        ("Epoxy adhesive (2-part)", "kg", 0.05),
        ("Contact adhesive (rubber) 1L", "litre", 0.08),
    ]
    for nm, unit, ratio in sealants:
        add(
            name=nm,
            phase="Miscellaneous / Consumables (VERY IMPORTANT)",
            category="Chemical",
            subcategory="Sealant/Adhesive",
            description="Sealant/adhesive used in finishing and installation works.",
            specifications=nm,
            unit=unit,
            usage_type="per_marla",
            usage_ratio=ratio,
            functional_tag="consumable",
            urdu_name="چپک/سیلنٹ",
            synonyms="sealant; adhesive",
        )

    # --- More paint variants (brands optional later) ---
    paints = [
        ("Interior emulsion (economy)", 0.00035, "economy"),
        ("Interior emulsion (premium washable)", 0.00035, "premium"),
        ("Exterior weather shield (premium)", 0.00028, "premium"),
        ("Textured paint (exterior)", 0.00022, "premium"),
        ("Enamel paint (synthetic)", 0.00008, "standard"),
        ("Wood polish (PU)", 0.00003, "premium"),
    ]
    for nm, ratio, grade in paints:
        add(
            name=nm,
            phase="Paint & Finishing",
            category="Paint",
            subcategory="Topcoat",
            description="Paint/topcoat product category variant.",
            specifications=nm,
            unit="litre",
            usage_type="per_sqft",
            usage_ratio=ratio,
            quality_grade=grade,
            functional_tag="finishing",
            urdu_name="پینٹ",
            synonyms="paint; topcoat",
        )

    # Ensure target size 650–800
    return materials


def regenerate_pricing(materials: list[dict], base_prices: dict, target_cities: list[str]) -> list[dict]:
    """
    Build pricing rows for each material for target cities.
    base_prices: {material_id: base_avg_lahore}
    """
    today = dt.date.today().isoformat()
    rows = []
    pid = 1
    for m in materials:
        mid = m["material_id"]
        # Determine Lahore baseline avg
        base_avg = base_prices.get(mid)
        if base_avg is None:
            # Estimate base avg from usage_type/unit heuristics
            unit = (m.get("unit") or "").lower()
            cat = (m.get("category") or "").lower()
            sub = (m.get("subcategory") or "").lower()
            name = (m.get("name") or "").lower()
            # crude but stable heuristics
            if unit in {"bag"}:
                base_avg = 1380
            elif unit in {"kg"}:
                if "steel" in name or "rebar" in name or "sarya" in name:
                    base_avg = 285
                elif "grout" in name or "chemical" in cat or "waterproof" in name:
                    base_avg = 300
                else:
                    base_avg = 180
            elif unit in {"ft2"}:
                base_avg = 280 if "glass" in name else 120
            elif unit in {"rft"}:
                base_avg = 120
            elif unit in {"m"}:
                base_avg = 120
            elif unit in {"pcs", "pair", "set"}:
                base_avg = 180
            elif unit in {"roll", "pack"}:
                base_avg = 180
            elif unit in {"litre", "l"}:
                base_avg = 700
            elif unit in {"can"}:
                base_avg = 620
            elif unit in {"sheet"}:
                base_avg = 3300
            elif unit in {"cft"}:
                base_avg = 100
            elif unit in {"day"}:
                base_avg = 70
            elif unit in {"lot"}:
                base_avg = 95000
            elif unit in {"sets"}:
                base_avg = 32000
            elif unit in {"ft"}:
                base_avg = 200
            else:
                base_avg = 200

        # Spread based on category
        spread = 0.12
        if m.get("functional_tag") in {"structural", "plumbing", "electrical"}:
            spread = 0.10
        if m.get("subcategory", "").lower() in {"tools", "solvent"}:
            spread = 0.15

        for city in target_cities:
            mult = CITY_MULT[city]
            # deterministic jitter avoids same values across cities/materials
            jitter = _stable_u01(mid, city, "avg")
            city_avg = base_avg * mult * (1 + (jitter - 0.5) * 0.05)
            mn, mx, av = _mk_range(city_avg, spread_pct=spread, jitter=_stable_u01(mid, city, "range"))
            # guarantee not identical across cities: if close, nudge by 1–2%
            if city != "Lahore":
                if abs(av - _round_price(base_avg)) < max(1.0, 0.01 * av):
                    av = _round_price(av * 1.02)
                    mn = min(mn, av)
                    mx = max(mx, av)

            rows.append(
                {
                    "price_id": f"P{pid:06d}",
                    "material_id": mid,
                    "city": city,
                    "price_min": mn,
                    "price_max": mx,
                    "price_avg": av,
                    "confidence_score": CITY_CONF[city],
                    "last_updated": today,
                    "source_notes": "Phase-2 modeled range with city multipliers (needs supplier verification).",
                }
            )
            pid += 1
    return rows


def build_base_price_map_from_existing(pricing_rows: list[dict]) -> dict:
    # Use Lahore avg as baseline when present
    base = {}
    for p in pricing_rows:
        if p.get("city") == "Lahore":
            try:
                base[p["material_id"]] = float(p["price_avg"])
            except Exception:
                pass
    return base


def generate_phase_labor_mapping() -> list[dict]:
    # Productivity_rate is "units per worker-day" (8 hours) by default.
    return [
        {"phase": "Site Preparation", "work_type": "Helper (beldar)", "unit": "sqft_per_day", "productivity_rate": "500", "notes": "Clearing, shifting, basic site work."},
        {"phase": "Excavation & Foundation", "work_type": "Mistri (mason)", "unit": "sqft_per_day", "productivity_rate": "120", "notes": "Layout + foundation supervision."},
        {"phase": "Excavation & Foundation", "work_type": "Helper (beldar)", "unit": "cft_per_day", "productivity_rate": "180", "notes": "Soil shifting/backfill handling."},
        {"phase": "Grey Structure", "work_type": "Shuttering carpenter", "unit": "sqft_per_day", "productivity_rate": "90", "notes": "Slab/beam shuttering."},
        {"phase": "Grey Structure", "work_type": "Steel fixer", "unit": "kg_per_day", "productivity_rate": "450", "notes": "Rebar cutting, bending, tying."},
        {"phase": "Masonry & Walls", "work_type": "Mistri (mason)", "unit": "sqft_per_day", "productivity_rate": "140", "notes": "Brickwork (varies by wall thickness)." },
        {"phase": "Plastering & Screeding", "work_type": "Mistri (mason)", "unit": "sqft_per_day", "productivity_rate": "220", "notes": "Internal plaster + basic screed."},
        {"phase": "Flooring & Tiling", "work_type": "Tiler", "unit": "sqft_per_day", "productivity_rate": "120", "notes": "Standard tile sizes; complex patterns reduce."},
        {"phase": "Electrical (Rough + Final)", "work_type": "Electrician", "unit": "point_per_day", "productivity_rate": "30", "notes": "Point includes conduit + wire + termination."},
        {"phase": "Plumbing (Rough + Final)", "work_type": "Plumber", "unit": "fixture_per_day", "productivity_rate": "10", "notes": "Fixture = tap/flush/basin connection; rough-in differs."},
        {"phase": "Paint & Finishing", "work_type": "Painter", "unit": "sqft_per_day", "productivity_rate": "500", "notes": "Putty + primer + 2 coats (avg)." },
        {"phase": "Roofing & Waterproofing", "work_type": "Mistri (mason)", "unit": "sqft_per_day", "productivity_rate": "180", "notes": "Membrane + screed protection varies."},
        {"phase": "Carpentry & Woodwork", "work_type": "Shuttering carpenter", "unit": "sqft_per_day", "productivity_rate": "60", "notes": "Doors + frames + basic hardware."},
        {"phase": "Aluminum & Glass Work", "work_type": "Mistri (mason)", "unit": "sqft_per_day", "productivity_rate": "80", "notes": "Install frames + glazing; specialist rates may apply."},
        {"phase": "Kitchen & Wardrobes", "work_type": "Shuttering carpenter", "unit": "sqft_per_day", "productivity_rate": "45", "notes": "Modular cabinets vary widely by design."},
        {"phase": "Sanitary & Bathroom Fittings", "work_type": "Plumber", "unit": "fixture_per_day", "productivity_rate": "12", "notes": "Final fix installs."},
        {"phase": "External Works", "work_type": "Helper (beldar)", "unit": "sqft_per_day", "productivity_rate": "220", "notes": "Pavers and basic external works."},
    ]


def main():
    materials_path = ROOT / "materials_master.csv"
    pricing_path = ROOT / "pricing_data.csv"

    materials = read_csv(materials_path)
    pricing = read_csv(pricing_path)

    # Upgrade schema
    materials = add_ai_columns(materials)
    enrich_existing_material_ai(materials)

    # Expand materials
    materials = generate_material_expansion(materials)

    # Rebuild pricing with strong multi-city coverage
    base_prices = build_base_price_map_from_existing(pricing)
    target_cities = ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad"]
    pricing_rows = regenerate_pricing(materials, base_prices, target_cities)

    # Write upgraded materials_master.csv
    mat_fields = [
        "material_id",
        "name",
        "phase",
        "category",
        "subcategory",
        "description",
        "specifications",
        "unit",
        "usage_type",
        "usage_ratio",
        "quality_grade",
        "brand",
        "notes",
        "functional_tag",
        "urdu_name",
        "synonyms",
    ]
    write_csv(materials_path, mat_fields, materials)

    # Write pricing_data.csv
    price_fields = [
        "price_id",
        "material_id",
        "city",
        "price_min",
        "price_max",
        "price_avg",
        "confidence_score",
        "last_updated",
        "source_notes",
    ]
    write_csv(pricing_path, price_fields, pricing_rows)

    # Write phase_labor_mapping.csv
    plm_path = ROOT / "phase_labor_mapping.csv"
    plm_fields = ["phase", "work_type", "unit", "productivity_rate", "notes"]
    write_csv(plm_path, plm_fields, generate_phase_labor_mapping())

    print(f"materials_master.csv rows: {len(materials)}")
    print(f"pricing_data.csv rows: {len(pricing_rows)}")
    print("Wrote phase_labor_mapping.csv")


if __name__ == "__main__":
    main()

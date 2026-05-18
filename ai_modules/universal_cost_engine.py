from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _waste_factor(floor_sqft: float) -> float:
    s = float(floor_sqft or 0)
    if s <= 250:
        return 1.15
    if s <= 500:
        return 1.12
    if s <= 900:
        return 1.10
    if s <= 1800:
        return 1.09
    return 1.08


def _sqft_per_marla(city: str) -> float:
    c = (city or "").strip().lower()
    # Spec: Lahore/Faisalabad/Multan use 225 sqft/marla, all others 272.
    if c in ("lahore", "faisalabad", "multan"):
        return 225.0
    return 272.0


def _classify_building_category(building_type: str) -> Tuple[Optional[str], List[str]]:
    t = (building_type or "").strip().lower()
    mapping = {
        "Residential": {"house", "apartment", "villa", "farmhouse", "servant_quarter"},
        "Commercial": {"shop", "plaza", "office", "hotel", "restaurant", "warehouse", "mall"},
        "Institutional": {
            "school",
            "university",
            "hospital",
            "clinic",
            "mosque",
            "library",
            "govt_office",
        },
        "Industrial": {"factory", "cold_storage", "workshop", "petrol_station"},
    }
    for cat, types in mapping.items():
        if t in types:
            return cat, []
    return None, [
        "⚠ [A1]: building_type is ambiguous or unsupported — provide an explicit building_type from the allowed list."
    ]


def _default_wall_height_ft(building_type: str) -> Optional[float]:
    t = (building_type or "").strip().lower()
    if t in ("house", "apartment", "villa", "farmhouse", "servant_quarter"):
        return 10.0
    if t in ("shop",):
        return 13.0
    if t in ("office", "plaza", "govt_office", "library", "university", "school"):
        return 12.0
    if t in ("hospital", "clinic"):
        return 13.5
    if t in ("mosque",):
        return 18.0
    if t in ("warehouse", "factory", "cold_storage", "workshop", "petrol_station"):
        return None  # must ask/receive if they want accuracy; we will warn + assume
    if t in ("hotel", "restaurant", "mall"):
        return 11.0
    return None


def _slab_thickness_in(building_category: str, building_type: str) -> float:
    t = (building_type or "").strip().lower()
    if t in ("hospital", "clinic"):
        return 6.0
    if building_category in ("Commercial", "Institutional"):
        return 5.0
    if building_category == "Industrial":
        # Unless user indicates heavy; default to 6in for safety in industrial
        return 6.0
    return 4.0


def _structural_system(building_category: str, building_type: str, floors: int, span_m: Optional[float], total_sqft: float) -> str:
    t = (building_type or "").strip().lower()
    if building_category in ("Commercial", "Institutional"):
        return "RCC frame"
    if building_category == "Residential":
        return "RCC frame"
    # Industrial selection
    if t == "warehouse" and total_sqft > 5000:
        return "PEB"
    if span_m is not None and float(span_m) > 15:
        return "Steel portal frame"
    return "RCC frame"


def _residential_layout_from_bhk(bhk: int) -> Tuple[int, int, int, int, List[str]]:
    # bedrooms, bathrooms, kitchens, extra_rooms, warnings
    b = int(bhk or 0)
    warnings: List[str] = []
    if b <= 0:
        return 0, 0, 0, 0, ["⚠ [A2]: bhk not provided for residential — provide bhk for room-derived geometry."]
    if b == 1:
        return 1, 1, 1, 0, []
    if b == 2:
        return 2, 2, 1, 0, []
    if b == 3:
        return 3, 2, 1, 1, []  # lounge
    if b == 4:
        return 4, 3, 1, 2, []  # lounge + dining
    if b == 5:
        return 5, 3, 2, 2, []  # 2 kitchens + lounge + dining
    # 6+
    return 6, 4, 2, 3, []  # lounge + dining + servant quarter


def _min_sqft_for_bhk(bhk: int) -> float:
    mins = {1: 225, 2: 450, 3: 675, 4: 900, 5: 1125}
    if bhk >= 6:
        return 1350.0
    return float(mins.get(int(bhk or 0), 0) or 0)


def _nonres_rooms_total(floor_sqft: float) -> int:
    # Bench: treat 1 room per 150 sqft, minimum 2.
    return max(2, int(math.ceil(float(floor_sqft) / 150.0)))


@dataclass
class _RateLookup:
    """
    Wraps CostEstimationModule price maps without importing pandas here.
    Callers should pass (get_rate_fn, get_marla_sqft_fn) or a dict-like adapter.
    """

    get_rate_pkr: Any  # (city_lower, query_key)->Optional[float]


def estimate_universal(
    *,
    building_type: str,
    city: str,
    area: Dict[str, Any],
    floors: Optional[int] = None,
    bhk: Optional[int] = None,
    finishing_tier: Optional[str] = None,
    capacity: Optional[Dict[str, Any]] = None,
    wall_height_ft: Optional[float] = None,
    span_m: Optional[float] = None,
    industrial_heavy: bool = False,
    rate_lookup: Optional[_RateLookup] = None,
) -> Dict[str, Any]:
    """
    Universal Pakistan construction cost engine (first-principles, no lump sums).
    Returns the required JSON schema.
    """
    capacity = dict(capacity or {})
    validation_warnings: List[str] = []
    layout_warnings: List[str] = []
    capacity_warnings: List[str] = []
    regulatory_flags: List[str] = []

    bcat, cat_w = _classify_building_category(building_type)
    if cat_w:
        # "Ask one clarifying question" is a UX behavior; API returns warnings.
        validation_warnings.extend(cat_w)
        bcat = ""

    t = (building_type or "").strip().lower()
    city_norm = (city or "").strip()
    city_lower = city_norm.lower()

    # ── Normalize area ────────────────────────────────────────────────────────
    sqft_per_marla = _sqft_per_marla(city_norm)
    floor_sqft: Optional[float] = None

    if "sqft" in area and area["sqft"] is not None:
        floor_sqft = float(area["sqft"])
    elif "marla" in area and area["marla"] is not None:
        floor_sqft = float(area["marla"]) * sqft_per_marla
    elif "kanal" in area and area["kanal"] is not None:
        floor_sqft = float(area["kanal"]) * 20.0 * sqft_per_marla
    elif "sqm" in area and area["sqm"] is not None:
        floor_sqft = float(area["sqm"]) * 10.76
    elif "sqy" in area and area["sqy"] is not None:
        floor_sqft = float(area["sqy"]) * 9.0
    elif "acre" in area and area["acre"] is not None:
        floor_sqft = float(area["acre"]) * 43560.0
    else:
        floor_sqft = None
        validation_warnings.append("⚠ [A3]: area is missing — provide one of sqft/marla/kanal/sqm/sqy/acre.")

    if floor_sqft is not None and floor_sqft <= 0:
        validation_warnings.append("⚠ [A4]: floor_sqft must be > 0.")

    # ── Default floors rules ─────────────────────────────────────────────────
    floors_norm: Optional[int] = int(floors) if floors is not None else None
    if floors_norm is None:
        if t in ("house", "warehouse", "factory", "mosque"):
            floors_norm = 1
        elif t in ("apartment", "shop", "hospital"):
            validation_warnings.append("⚠ [A5]: floors is required for this building_type — provide floors.")
            floors_norm = None
        else:
            validation_warnings.append("⚠ [A6]: floors omitted and ambiguous — provide floors.")
            floors_norm = None

    if floors_norm is None:
        floors_norm = 1  # proceed with a safe default but warn above

    total_built_sqft = float(floor_sqft or 0) * float(floors_norm or 1)

    # ── Capacity-based area derivation (institutional) ───────────────────────
    if t == "hospital" and capacity.get("beds_count") and (floor_sqft is None or floor_sqft <= 0):
        floor_sqft = float(capacity["beds_count"]) * 350.0
        total_built_sqft = floor_sqft * float(floors_norm)
    if t == "school" and capacity.get("student_capacity"):
        cap = float(capacity["student_capacity"])
        derived = cap * 15.0
        if floor_sqft is None or floor_sqft <= 0:
            floor_sqft = derived
        else:
            floor_sqft = max(float(floor_sqft), derived)
        total_built_sqft = float(floor_sqft) * float(floors_norm)
    if t == "mosque" and capacity.get("worshipper_capacity"):
        cap = float(capacity["worshipper_capacity"])
        derived = cap * 9.0
        if floor_sqft is None or floor_sqft <= 0:
            floor_sqft = derived
        else:
            floor_sqft = max(float(floor_sqft), derived)
        total_built_sqft = float(floor_sqft) * float(floors_norm)

    if floor_sqft is None:
        floor_sqft = 0.0

    # ── Finishing tier normalization ─────────────────────────────────────────
    tier = (finishing_tier or "").strip() or "Standard"
    tier_norm = tier.strip().lower()
    # For industrial, spec says omit finish tier & finish items
    if bcat == "Industrial":
        tier = ""

    # ── Rooms / openings ─────────────────────────────────────────────────────
    bedrooms = bathrooms = kitchens = extra_rooms = 0
    rooms_total = 0
    occupants = None
    is_residential = t in ("house", "apartment", "villa", "farmhouse", "servant_quarter")
    if is_residential:
        b_in = int(bhk or 0)
        bedrooms, bathrooms, kitchens, extra_rooms, ws = _residential_layout_from_bhk(b_in)
        layout_warnings.extend(ws)
        rooms_total = bedrooms + bathrooms + kitchens + extra_rooms
        occupants = bedrooms * 2 + 1 if bedrooms else None
        min_sqft = _min_sqft_for_bhk(b_in)
        if min_sqft and float(floor_sqft) < min_sqft:
            layout_warnings.append(
                f"⚠ [L1]: BHK exceeds area capacity — got floor_sqft={int(floor_sqft)}, expected >= {int(min_sqft)} for {b_in} BHK."
            )
    else:
        rooms_total = _nonres_rooms_total(float(floor_sqft))

    # Door & window counts
    if is_residential and bedrooms:
        door_count = bedrooms + bathrooms + kitchens + 1 + (extra_rooms // 2)
        window_count = int(math.ceil(bedrooms * 1.5 + kitchens + (1 if extra_rooms else 0)))
    else:
        service_doors = int(math.ceil(float(floor_sqft) / 500.0))
        door_count = rooms_total + 1 + service_doors
        window_count = int(math.ceil(float(floor_sqft) / 80.0)) if floor_sqft else 0

    # ── Geometry ─────────────────────────────────────────────────────────────
    perimeter = 4.0 * math.sqrt(float(floor_sqft)) if floor_sqft else 0.0
    wall_h = wall_height_ft if wall_height_ft is not None else _default_wall_height_ft(building_type)
    if wall_h is None:
        # Spec says ASK for warehouse heights; API returns warning + default assumption.
        wall_h = 22.0
        validation_warnings.append(
            "⚠ [A7]: wall_height_ft not provided for this building_type — assumed 22 ft. Provide wall_height_ft for accuracy."
        )

    internal_partition_len = 0.0
    if rooms_total > 0 and floor_sqft:
        internal_partition_len = (rooms_total - 1) * math.sqrt(float(floor_sqft) / float(rooms_total)) * 1.5

    gross_wall_area = (perimeter + internal_partition_len) * wall_h * float(floors_norm)
    opening_area = door_count * 21.0 + window_count * 12.0
    net_wall_area = max(0.0, gross_wall_area - opening_area)
    plaster_area = net_wall_area * 2.0 + float(floor_sqft) * float(floors_norm)

    slab_th_in = 8.0 if (industrial_heavy and bcat == "Industrial") else _slab_thickness_in(bcat or "", building_type)
    slab_th_ft = slab_th_in / 12.0
    slab_conc_vol_cft = float(floor_sqft) * float(floors_norm) * slab_th_ft

    column_spacing_ft = 10.0  # per spec for formula
    col_count = int(math.ceil((perimeter + internal_partition_len) / column_spacing_ft)) if (perimeter + internal_partition_len) else 0

    # Residential tile areas
    floor_tile_area = None
    wet_tile_area = None
    if is_residential and rooms_total > 0:
        floor_tile_area = float(floor_sqft) * float(floors_norm) - bathrooms * 50.0
        wet_tile_area = bathrooms * (50.0 + 120.0)

    # Electrical points (residential only by spec)
    total_points = None
    if is_residential and bedrooms:
        points = (
            bedrooms * 6
            + bathrooms * 2
            + kitchens * 6
            + (1 * 8 if extra_rooms else 0)  # lounge
            + (1 * 3 if extra_rooms >= 2 else 0)  # dining
            + 1  # stair/lobby
        )
        total_points = int(points)

    wf = _waste_factor(float(floor_sqft))

    # ── Structural selection & core validation ───────────────────────────────
    ss = _structural_system(bcat or "", building_type, floors_norm, span_m, total_built_sqft)
    if floors_norm > 2 and ss != "RCC frame":
        validation_warnings.append("⚠ [S6]: floors > 2 → structural_system must be RCC frame.")
        ss = "RCC frame"

    # ── Grey structure quantities (5A) ───────────────────────────────────────
    wall_brick_vol = net_wall_area * 0.75
    foundation_brick_vol = perimeter * 4.0 * 1.125
    parapet_vol = perimeter * 3.0 * 0.75
    brick_count = (wall_brick_vol + foundation_brick_vol + parapet_vol) * 13.5 * wf

    masonry_cement = (brick_count / 500.0) * 1.5
    lean_pcc_cement = (float(floor_sqft) * 0.5 / 27.0) * 1.5
    column_cement = float(col_count) * 0.5 * 5.0
    slab_cement = (float(floor_sqft) * float(floors_norm) * slab_th_ft / 27.0) * 6.5
    plaster_cement = plaster_area * 0.012
    total_cement = (masonry_cement + lean_pcc_cement + column_cement + slab_cement + plaster_cement) * wf

    masonry_sand = brick_count * 0.04
    lean_sand = float(floor_sqft) * 0.5 * 0.44
    slab_sand = float(floor_sqft) * float(floors_norm) * slab_th_ft * 0.44
    plaster_sand = plaster_area * 0.025
    column_sand = float(col_count) * 0.5 * 0.44
    total_sand = (masonry_sand + lean_sand + slab_sand + plaster_sand + column_sand) * wf

    lean_crush = float(floor_sqft) * 0.5 / 27.0 * 0.88
    slab_crush = float(floor_sqft) * float(floors_norm) * slab_th_ft * 0.88
    column_crush = float(col_count) * 0.5 * 0.88
    total_crush = (lean_crush + slab_crush + column_crush) * wf

    slab_steel = float(floor_sqft) * float(floors_norm) * slab_th_ft * 2.4
    column_steel = float(col_count) * 2.5
    foundation_ties = perimeter * 1.2
    stirrups = float(col_count) * 0.8
    total_steel = (slab_steel + column_steel + foundation_ties + stirrups) * wf

    shuttering_sheets = int(math.ceil((float(floor_sqft) * float(floors_norm)) / 32.0)) if floor_sqft else 0
    shuttering_labour = float(floor_sqft) * float(floors_norm) * 50.0
    slab_placement_labour = slab_conc_vol_cft * 90.0

    # Plaster can be omitted only for industrial warehouse
    plaster_labour = plaster_area * 42.0
    if t in ("warehouse",) and bcat == "Industrial":
        # still keep structure items, but warn + omit plastering
        validation_warnings.append("⚠ [S5]: plastering omitted for industrial warehouse (allowed exception).")
        plaster_labour = 0.0

    excavation_vol = perimeter * 2.5 * 5.0
    excavation_labour = excavation_vol * 22.0
    earth_bags = int(math.ceil(excavation_vol * 0.037))

    # ── Pricing: rate lookup adapter (best effort) ───────────────────────────
    def rate(key: str) -> Optional[float]:
        if not rate_lookup:
            return None
        try:
            r = rate_lookup.get_rate_pkr(city_lower, key)
            return float(r) if r is not None and float(r) > 0 else None
        except Exception:
            return None

    # Keys are logical; the module can map them to materials_master/pricing_data.
    rates = {
        "brick": rate("brick"),
        "cement_bag": rate("cement_bag"),
        "sand_cft": rate("sand_cft"),
        "crush_cft": rate("crush_cft"),
        "steel_kg": rate("steel_kg"),
    }

    def _line(item: str, qty: float, unit: str, rate_pkr: Optional[float], wf_applied: Optional[float], formula: str) -> Dict[str, Any]:
        rp = float(rate_pkr) if rate_pkr is not None else None
        subtotal = float(qty) * rp if (rp is not None) else None
        return {
            "item": item,
            "qty": float(qty),
            "unit": unit,
            "rate_pkr": rp,
            "subtotal_pkr": subtotal,
            "waste_factor_applied": wf_applied,
            "formula_note": formula,
        }

    categories: List[Dict[str, Any]] = []

    grey_items: List[Dict[str, Any]] = [
        _line("Bricks (count)", brick_count, "nos", rates["brick"], wf, "brick_count=(wall+foundation+parapet)cft*13.5*waste"),
        _line("Cement (bags)", total_cement, "bags", rates["cement_bag"], wf, "total_cement=(masonry+lean+column+slab+plaster)*waste"),
        _line("Sand (cft)", total_sand, "cft", rates["sand_cft"], wf, "total_sand=(masonry+lean+slab+plaster+column)*waste"),
        _line("Crush (cft)", total_crush, "cft", rates["crush_cft"], wf, "total_crush=(lean+slab+column)*waste"),
        _line("Steel Rebar (kg)", total_steel, "kg", rates["steel_kg"], wf, "total_steel=(slab+column+ties+stirrups)*waste"),
        _line("Slab concrete volume", slab_conc_vol_cft, "cft", None, None, "slab_concrete_vol=floor_sqft*floors*slab_thickness_ft"),
        _line("Shuttering sheets", shuttering_sheets, "sheets", None, None, "ceil(floor_sqft*floors/32)"),
        _line("Shuttering labour", float(floor_sqft) * float(floors_norm), "sqft", 50.0, None, "floor_sqft*floors*Rs50/sqft"),
        _line("Slab placement labour", slab_conc_vol_cft, "cft", 90.0, None, "slab_concrete_vol*Rs90/cft"),
        _line("Plaster labour", plaster_area, "sqft", 42.0 if plaster_labour else 0.0, None, "plaster_area*Rs42/sqft"),
        _line("Excavation volume", excavation_vol, "cft", None, None, "perimeter*2.5*5"),
        _line("Excavation labour", excavation_vol, "cft", 22.0, None, "excavation_vol*Rs22/cft"),
        _line("Earth bags", earth_bags, "bags", None, None, "ceil(excavation_vol*0.037)"),
    ]
    grey_total = sum(x["subtotal_pkr"] for x in grey_items if x["subtotal_pkr"] is not None)
    categories.append({"name": "Grey Structure", "category_type": "core", "total_pkr": grey_total, "items": grey_items})

    # ── Finishing (residential only detailed; others benchmark warn) ──────────
    finishing_total = 0.0
    finishing_items: List[Dict[str, Any]] = []
    if bcat != "Industrial":
        if is_residential and floor_tile_area is not None and wet_tile_area is not None:
            tile_waste = max(1.15, wf)
            if tier_norm in ("standard", "economy", ""):
                floor_rate = 135.0
                bath_floor_rate = 165.0
                bath_wall_rate = 145.0
            elif tier_norm == "premium":
                floor_rate = 280.0
                bath_floor_rate = 300.0
                bath_wall_rate = 260.0
            else:  # luxury
                floor_rate = 550.0
                bath_floor_rate = 550.0
                bath_wall_rate = 550.0
            finishing_items.extend(
                [
                    _line("Floor tiles", floor_tile_area * tile_waste, "sqft", floor_rate, tile_waste, "floor_tile_area*waste"),
                    _line("Bathroom floor tiles", bathrooms * 50.0 * tile_waste, "sqft", bath_floor_rate, tile_waste, "bathrooms*50*waste"),
                    _line("Bathroom wall tiles", bathrooms * 120.0 * tile_waste, "sqft", bath_wall_rate, tile_waste, "bathrooms*120*waste"),
                    _line("Tile adhesive + grout", (floor_tile_area + wet_tile_area) * tile_waste, "sqft", 25.0, tile_waste, "(floor+wet)*Rs25"),
                    _line("Tile labour", (floor_tile_area + wet_tile_area) * tile_waste, "sqft", 55.0, tile_waste, "(floor+wet)*Rs55"),
                ]
            )

            # Doors/windows pricing from spec
            if tier_norm in ("premium",):
                main_door = 70000.0
                interior = 45000.0
                bath_door = 27000.0
                window_rate = 22000.0
                grille = 0.0
            elif tier_norm in ("luxury",):
                main_door = 180000.0
                interior = 90000.0
                bath_door = 55000.0
                window_rate = 55000.0
                grille = 0.0
            else:
                main_door = 33000.0
                interior = 21000.0
                bath_door = 14000.0
                window_rate = 10000.0
                grille = 3500.0

            finishing_items.extend(
                [
                    _line("Main door", 1, "nos", main_door, None, "1 main door"),
                    _line("Interior doors", max(0, bedrooms + (1 if extra_rooms else 0)), "nos", interior, None, "beds + lounge"),
                    _line("Bathroom doors", bathrooms, "nos", bath_door, None, "bathrooms"),
                    _line("Door frames + fitting", door_count, "nos", 3500.0, None, "door_count*Rs3500"),
                    _line("Windows", window_count, "nos", window_rate, None, "window_count*tier rate"),
                    _line("Window fitting", window_count, "nos", 1200.0, None, "window_count*Rs1200"),
                ]
            )
            if grille:
                finishing_items.append(_line("Window grilles (Std)", window_count, "nos", grille, None, "window_count*Rs3500"))

            # Paint
            interior_paintable = max(0.0, plaster_area - wet_tile_area)
            if tier_norm == "premium":
                in_rate = 105.0
                ex_rate = 95.0
            elif tier_norm == "luxury":
                in_rate = 195.0
                ex_rate = 95.0
            else:
                in_rate = 62.0
                ex_rate = 52.0
            finishing_items.extend(
                [
                    _line("Interior paint", interior_paintable, "sqft", in_rate, None, "plaster_area-wet_tile_area"),
                    _line("Exterior paint", perimeter * wall_h, "sqft", ex_rate, None, "perimeter*wall_height"),
                ]
            )

            # Sanitary allowance (per bathroom)
            if tier_norm == "premium":
                sanitary_per = 60000.0
            elif tier_norm == "luxury":
                sanitary_per = 180000.0
            else:
                sanitary_per = 26000.0
            finishing_items.append(_line("Sanitary fixtures allowance", bathrooms, "bathroom", sanitary_per, None, "bathrooms*allowance"))

        else:
            # Non-residential finishing benchmarks are minimum guards; we still avoid lump sums by returning as a warning item.
            validation_warnings.append(
                "⚠ [F0]: Non-residential finishing is not room-detailed in this endpoint yet — provide room program inputs to derive finish quantities, or accept benchmark-only warnings."
            )

    finishing_total = sum(x["subtotal_pkr"] for x in finishing_items if x["subtotal_pkr"] is not None)
    if finishing_items:
        categories.append({"name": "Finishing", "category_type": "core", "total_pkr": finishing_total, "items": finishing_items})

    # ── Electrical (residential detailed) ─────────────────────────────────────
    electrical_total = 0.0
    electrical_items: List[Dict[str, Any]] = []
    if is_residential and total_points is not None:
        switches_sockets = int(math.ceil(total_points * 1.1))
        mcb_count = int(math.ceil(total_points / 6.0) + 2)
        db_ways = int(max(8, mcb_count + 2))

        avg_run = math.sqrt(float(floor_sqft)) * 0.6 + wall_h * 0.3 if floor_sqft else 0.0
        total_wire_m = float(total_points) * avg_run * 1.15
        conduit_ft = total_wire_m * 3.28 * 0.85
        junction_boxes = float(total_points) * 0.8

        electrical_items.extend(
            [
                _line("Electrical labour", total_built_sqft, "sqft", 62.0 if tier_norm in ("standard", "economy", "") else 95.0, None, "total_built_sqft*labour_rate"),
                _line("Switches + sockets (count)", switches_sockets, "nos", None, None, "ceil(total_points*1.1)"),
                _line("MCB count", mcb_count, "nos", None, None, "ceil(points/6)+2"),
                _line("DB ways", db_ways, "ways", None, None, "max(8, MCB+2)"),
                _line("Wiring length", total_wire_m, "m", None, None, "points*avg_run*1.15"),
                _line("Conduit length", conduit_ft, "ft", None, None, "wire_m*3.28*0.85"),
                _line("Junction boxes", junction_boxes, "nos", None, None, "points*0.8"),
            ]
        )
        electrical_total = sum(x["subtotal_pkr"] for x in electrical_items if x["subtotal_pkr"] is not None)
        categories.append({"name": "Electrical", "category_type": "core", "total_pkr": electrical_total, "items": electrical_items})

        # Validation M1/M2
        if switches_sockets > total_points * 1.3:
            validation_warnings.append(f"⚠ [M1]: switches_sockets too high — got {switches_sockets}, expected ≤ {total_points*1.3:.1f}")
        if switches_sockets < total_points * 0.8:
            validation_warnings.append(f"⚠ [M2]: switches_sockets too low — got {switches_sockets}, expected ≥ {total_points*0.8:.1f}")

    # ── Plumbing (residential detailed) ───────────────────────────────────────
    plumbing_total = 0.0
    plumbing_items: List[Dict[str, Any]] = []
    if is_residential and bedrooms:
        pprc_ft = bathrooms * 35 + kitchens * 20 + floors_norm * 15
        upvc_ft = bathrooms * 20 + kitchens * 15 + floors_norm * 10
        gas_ft = kitchens * 30 + floors_norm * 10
        fittings = (pprc_ft / 5.0) * 1.5
        ball_valves = bathrooms * 2 + kitchens
        floor_traps = bathrooms + kitchens * 0.5
        tank_l = max(500, int((occupants or 0) * 300))
        plumbing_items.extend(
            [
                _line("Plumbing labour", total_built_sqft, "sqft", 62.0, None, "total_built_sqft*Rs62"),
                _line("PPRC hot/cold pipe", pprc_ft, "ft", None, None, "bathrooms*35 + kitchens*20 + floors*15"),
                _line("UPVC sewer 4-inch", upvc_ft, "ft", None, None, "bathrooms*20 + kitchens*15 + floors*10"),
                _line("GI gas pipe", gas_ft, "ft", None, None, "kitchens*30 + floors*10"),
                _line("PPRC fittings", fittings, "nos", None, None, "PPRC_ft/5*1.5"),
                _line("Ball valves", ball_valves, "nos", None, None, "bathrooms*2 + kitchens"),
                _line("Floor traps", floor_traps, "nos", None, None, "bathrooms + kitchens*0.5"),
                _line("Water tank capacity", tank_l, "litres", None, None, "max(500, occupants*300)"),
            ]
        )
        plumbing_total = sum(x["subtotal_pkr"] for x in plumbing_items if x["subtotal_pkr"] is not None)
        categories.append({"name": "Plumbing", "category_type": "core", "total_pkr": plumbing_total, "items": plumbing_items})

        # Septic validation M3
        if occupants is not None and occupants <= 4:
            # we only emit as regulatory/validation note; costing depends on site
            validation_warnings.append("⚠ [M3]: For ≤4 occupants, septic should be < Rs 40,000 (precast rings).")

    # ── Mandatory validations (S1.., CS..) ───────────────────────────────────
    if brick_count > 2500.0 * float(floors_norm):
        validation_warnings.append(
            f"⚠ [S1]: brick_count high — got {int(brick_count)}, expected ≤ {int(2500*floors_norm)}"
        )
    if total_cement > 40.0 * float(floors_norm):
        validation_warnings.append(
            f"⚠ [S2]: total_cement_bags high — got {int(total_cement)}, expected ≤ {int(40*floors_norm)}"
        )
    if total_sand > 100.0 * float(floors_norm):
        validation_warnings.append(
            f"⚠ [S3]: total_sand_cft high — got {int(total_sand)}, expected ≤ {int(100*floors_norm)}"
        )
    # S4 slab mandatory: always present; keep check anyway
    if slab_conc_vol_cft <= 0:
        validation_warnings.append("⚠ [S4]: slab_concrete missing/zero — slab is mandatory.")
    if not (t == "warehouse" and bcat == "Industrial") and plaster_area <= 0:
        validation_warnings.append("⚠ [S5]: plastering missing/zero — plaster is mandatory except industrial warehouse.")
    if t in ("hospital", "clinic") and slab_th_in < 6.0:
        validation_warnings.append("⚠ [S7]: hospital → slab_thickness must be ≥ 6 in.")

    # Finish floor guard (F1) for residential benchmark
    if is_residential and bcat != "Industrial":
        min_rate = 800.0  # spec minimum benchmark for residential standard
        if tier_norm == "premium":
            min_rate = 1200.0
        if tier_norm == "luxury":
            min_rate = 1800.0
        if finishing_total and finishing_total < min_rate * float(total_built_sqft):
            validation_warnings.append(
                f"⚠ [F1]: finishing_total below minimum guard — got {int(finishing_total)}, expected ≥ {int(min_rate*total_built_sqft)}"
            )

    # Cost sanity (CS1): only if we have some priced totals
    total_pkr = grey_total + finishing_total + electrical_total + plumbing_total
    cost_per_sqft = (total_pkr / total_built_sqft) if (total_built_sqft and total_pkr) else None
    if cost_per_sqft is not None:
        if is_residential and tier_norm in ("standard", "economy", ""):
            if city_lower in ("lahore", "faisalabad", "multan"):
                lo, hi = 2400.0, 6000.0
            else:
                lo, hi = 3000.0, 7500.0
            if not (lo <= cost_per_sqft <= hi):
                validation_warnings.append(
                    f"⚠ [CS1]: cost_per_sqft out of benchmark — got {cost_per_sqft:.0f}, expected {int(lo)}–{int(hi)}"
                )

    # Regulatory flags
    if t in ("hospital", "school", "factory", "warehouse"):
        regulatory_flags.append(
            "⚠ Regulatory: NOC/approval from relevant authority (e.g., PHSA/Punjab Health, Ministry of Education, EOBI/Labour Dept., local municipal authority) is mandatory before construction."
        )

    # Rate lookup missing warnings
    for k, v in rates.items():
        if v is None and k in ("cement_bag", "steel_kg", "sand_cft", "crush_cft", "brick"):
            validation_warnings.append(f"⚠ [P1]: Missing rate for {k} in city='{city_norm}' — totals may be undercounted.")

    excluded_items = [
        "Land cost",
        "Architect/structural engineer fees (5–8% of construction)",
        "Utility connection charges",
        "Furniture, equipment, loose fixtures",
        "Contractor profit margin (15–25%)",
    ]

    # Mandatory disclaimer (always append)
    validation_warnings.append(
        "⚠ Exclusions: Land cost, architect/structural engineer fees (5–8% of construction), utility connection charges, furniture, equipment, loose fixtures, and contractor profit margin (15–25%) are NOT included in this estimate."
    )
    validation_warnings.append(
        "⚠ Accuracy: All quantities are derived from area-based benchmarks and geometric approximations. Commission engineer-stamped drawings for procurement-grade BOQ quantities."
    )

    return {
        "project": {
            "building_type": building_type,
            "building_category": bcat or "",
            "city": city_norm,
            "marla": float(area["marla"]) if isinstance(area.get("marla"), (int, float)) else None,
            "sqft_per_marla": float(sqft_per_marla),
            "floor_sqft": float(floor_sqft),
            "total_built_sqft": float(total_built_sqft),
            "floors": int(floors_norm),
            "bhk": int(bhk) if bhk is not None else None,
            "rooms_total": int(rooms_total) if rooms_total else None,
            "occupants": int(occupants) if occupants is not None else None,
            "finishing_tier": finishing_tier or "",
            "structural_system": ss,
            "capacity": dict(capacity or {}),
        },
        "derived_geometry": {
            "perimeter_ft": float(perimeter),
            "wall_height_ft": float(wall_h),
            "slab_thickness_in": float(slab_th_in),
            "internal_partition_len_ft": float(internal_partition_len),
            "gross_wall_area_sqft": float(gross_wall_area),
            "net_wall_area_sqft": float(net_wall_area),
            "plaster_area_sqft": float(plaster_area),
            "door_count": int(door_count),
            "window_count": int(window_count),
            "floor_tile_area_sqft": float(floor_tile_area) if floor_tile_area is not None else None,
            "wet_tile_area_sqft": float(wet_tile_area) if wet_tile_area is not None else None,
            "column_count": int(col_count),
            "column_spacing_ft": float(column_spacing_ft),
            "total_electrical_points": int(total_points) if total_points is not None else None,
            "waste_factor": float(wf),
        },
        "categories": categories,
        "summary": {
            "grey_structure_pkr": float(grey_total),
            "finishing_pkr": float(finishing_total),
            "electrical_pkr": float(electrical_total),
            "plumbing_pkr": float(plumbing_total),
            "hvac_pkr": None,
            "specialised_systems_pkr": None,
            "misc_pkr": 0.0,
            "total_pkr": float(total_pkr),
            "cost_per_sqft": float(cost_per_sqft) if cost_per_sqft is not None else None,
        },
        "excluded_items": excluded_items,
        "validation_warnings": validation_warnings,
        "layout_warnings": layout_warnings,
        "capacity_warnings": capacity_warnings,
        "regulatory_flags": regulatory_flags,
    }


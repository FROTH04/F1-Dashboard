"""
Internal helper: generates f1_data_2026_demo.json.
Run once from project root: python scripts/_gen_demo_data.py
"""
import json, random, math
from pathlib import Path

ROOT = Path(__file__).parent.parent
random.seed(42)

# ── Driver / team meta ────────────────────────────────────────────────────────
DRIVERS = [
    {"code": "ANT", "name": "Andrea Kimi Antonelli", "team": "Mercedes",     "number": 12, "color": "#27F4D2"},
    {"code": "RUS", "name": "George Russell",         "team": "Mercedes",     "number": 63, "color": "#27F4D2"},
    {"code": "HAM", "name": "Lewis Hamilton",          "team": "Ferrari",      "number": 44, "color": "#E8002D"},
    {"code": "LEC", "name": "Charles Leclerc",         "team": "Ferrari",      "number": 16, "color": "#E8002D"},
    {"code": "NOR", "name": "Lando Norris",            "team": "McLaren",      "number":  4, "color": "#FF8000"},
    {"code": "PIA", "name": "Oscar Piastri",           "team": "McLaren",      "number": 81, "color": "#FF8000"},
    {"code": "VER", "name": "Max Verstappen",          "team": "Red Bull",     "number":  1, "color": "#3671C6"},
    {"code": "HAD", "name": "Isack Hadjar",            "team": "Red Bull",     "number":  6, "color": "#3671C6"},
    {"code": "GAS", "name": "Pierre Gasly",            "team": "Alpine",       "number": 10, "color": "#FF69B4"},
    {"code": "COL", "name": "Franco Colapinto",        "team": "Alpine",       "number": 43, "color": "#FF69B4"},
    {"code": "LAW", "name": "Liam Lawson",             "team": "Racing Bulls", "number": 30, "color": "#6692FF"},
    {"code": "LIN", "name": "Arvid Lindblad",          "team": "Racing Bulls", "number": 21, "color": "#6692FF"},
    {"code": "BEA", "name": "Oliver Bearman",          "team": "Haas",         "number": 87, "color": "#B6BABD"},
    {"code": "OCO", "name": "Esteban Ocon",            "team": "Haas",         "number": 31, "color": "#B6BABD"},
    {"code": "BOR", "name": "Gabriel Bortoleto",       "team": "Audi",         "number":  5, "color": "#F50000"},
    {"code": "HUL", "name": "Nico Hülkenberg",         "team": "Audi",         "number": 27, "color": "#F50000"},
    {"code": "SAI", "name": "Carlos Sainz",            "team": "Williams",     "number": 55, "color": "#64C4FF"},
    {"code": "ALB", "name": "Alexander Albon",         "team": "Williams",     "number": 23, "color": "#64C4FF"},
    {"code": "BOT", "name": "Valtteri Bottas",         "team": "Cadillac",     "number": 77, "color": "#6B9B37"},
    {"code": "PER", "name": "Sergio Pérez",            "team": "Cadillac",     "number": 11, "color": "#6B9B37"},
    {"code": "ALO", "name": "Fernando Alonso",         "team": "Aston Martin", "number": 14, "color": "#358C75"},
    {"code": "STR", "name": "Lance Stroll",            "team": "Aston Martin", "number": 18, "color": "#358C75"},
]
DRIVER_INFO = {d["code"]: d for d in DRIVERS}
ALL_CODES = [d["code"] for d in DRIVERS]

PTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

# ── Race calendar ─────────────────────────────────────────────────────────────
# Official round numbers 1-9; rounds 4 & 5 (Bahrain, Saudi) were cancelled.
# We use sequential 1-9 for demo simplicity.
CALENDAR = [
    {"round": 1, "name": "Australian Grand Prix", "circuit": "Albert Park",           "country": "Australia", "date": "2026-03-15", "laps": 58,  "base_lap": 81.2,  "s1": 24.1, "s2": 33.8, "s3": 23.3},
    {"round": 2, "name": "Chinese Grand Prix",    "circuit": "Shanghai International", "country": "China",     "date": "2026-03-22", "laps": 56,  "base_lap": 93.5,  "s1": 28.2, "s2": 39.6, "s3": 25.7},
    {"round": 3, "name": "Japanese Grand Prix",   "circuit": "Suzuka",                 "country": "Japan",     "date": "2026-04-05", "laps": 53,  "base_lap": 89.7,  "s1": 26.9, "s2": 37.4, "s3": 25.4},
    {"round": 4, "name": "Miami Grand Prix",      "circuit": "Miami International",    "country": "USA",       "date": "2026-05-03", "laps": 57,  "base_lap": 90.1,  "s1": 27.0, "s2": 38.2, "s3": 24.9},
    {"round": 5, "name": "Monaco Grand Prix",     "circuit": "Circuit de Monaco",      "country": "Monaco",    "date": "2026-05-25", "laps": 78,  "base_lap": 72.5,  "s1": 22.1, "s2": 29.8, "s3": 20.6},
    {"round": 6, "name": "Canadian Grand Prix",   "circuit": "Circuit Gilles Villeneuve","country": "Canada",  "date": "2026-06-08", "laps": 70,  "base_lap": 73.6,  "s1": 22.3, "s2": 30.4, "s3": 20.9},
    {"round": 7, "name": "Austrian Grand Prix",   "circuit": "Red Bull Ring",          "country": "Austria",   "date": "2026-06-21", "laps": 71,  "base_lap": 65.8,  "s1": 19.8, "s2": 27.1, "s3": 18.9},
    {"round": 8, "name": "British Grand Prix",    "circuit": "Silverstone",            "country": "UK",        "date": "2026-07-05", "laps": 52,  "base_lap": 88.4,  "s1": 26.5, "s2": 37.0, "s3": 24.9},
    {"round": 9, "name": "Spanish Grand Prix",    "circuit": "Circuit de Barcelona-Catalunya","country": "Spain","date": "2026-06-29","laps": 66,  "base_lap": 79.5,  "s1": 23.8, "s2": 33.4, "s3": 22.3},
]

# ── Finishing orders per race ─────────────────────────────────────────────────
# Tuple: (code, dnf_flag)  — first entry is race winner
# ANT DNFs in R3 (car failure from P1 lead) and R9 (Barcelona — DNF from lead)
RACE_ORDERS = {
    1: [("ANT",False),("RUS",False),("LEC",False),("HAM",False),("NOR",False),("PIA",False),("VER",False),("HAD",False),("GAS",False),("COL",False),
        ("LAW",False),("LIN",False),("BEA",False),("SAI",False),("ALB",False),("HUL",False),("BOR",False),("OCO",False),("ALO",False),("BOT",False),("PER",False),("STR",True)],
    2: [("ANT",False),("RUS",False),("HAM",False),("LEC",False),("NOR",False),("PIA",False),("VER",False),("HAD",False),("LAW",False),("GAS",False),
        ("COL",False),("BEA",False),("SAI",False),("ALB",False),("LIN",False),("HUL",False),("BOR",False),("OCO",False),("ALO",False),("BOT",False),("STR",False),("PER",True)],
    3: [("LEC",False),("RUS",False),("HAM",False),("NOR",False),("PIA",False),("VER",False),("HAD",False),("GAS",False),("LAW",False),("COL",False),
        ("LIN",False),("BEA",False),("SAI",False),("ALB",False),("HUL",False),("BOR",False),("OCO",False),("ALO",False),("BOT",False),("STR",False),("PER",False),("ANT",True)],   # ANT DNF #1
    4: [("ANT",False),("HAM",False),("LEC",False),("RUS",False),("NOR",False),("PIA",False),("VER",False),("HAD",False),("GAS",False),("LAW",False),
        ("COL",False),("LIN",False),("BEA",False),("SAI",False),("ALB",False),("HUL",False),("BOR",False),("OCO",False),("ALO",False),("BOT",False),("STR",False),("PER",True)],
    5: [("RUS",False),("LEC",False),("HAM",False),("ANT",False),("NOR",False),("VER",False),("PIA",False),("GAS",False),("HAD",False),("COL",False),
        ("LAW",False),("LIN",False),("BEA",False),("SAI",False),("ALB",False),("HUL",False),("BOR",False),("OCO",False),("ALO",False),("BOT",False),("STR",False),("PER",True)],
    6: [("ANT",False),("RUS",False),("LEC",False),("HAM",False),("NOR",False),("VER",False),("PIA",False),("HAD",False),("GAS",False),("BEA",False),
        ("COL",False),("LIN",False),("LAW",False),("SAI",False),("ALB",False),("HUL",False),("BOR",False),("OCO",False),("ALO",False),("BOT",False),("STR",False),("PER",True)],
    7: [("ANT",False),("RUS",False),("LEC",False),("HAM",False),("NOR",False),("PIA",False),("VER",False),("HAD",False),("GAS",False),("LAW",False),
        ("COL",False),("LIN",False),("BEA",False),("SAI",False),("ALB",False),("HUL",False),("BOR",False),("OCO",False),("ALO",False),("BOT",False),("STR",False),("PER",True)],
    8: [("RUS",False),("ANT",False),("HAM",False),("LEC",False),("NOR",False),("PIA",False),("VER",False),("HAD",False),("GAS",False),("COL",False),
        ("LIN",False),("LAW",False),("BEA",False),("SAI",False),("ALB",False),("HUL",False),("BOR",False),("OCO",False),("ALO",False),("BOT",False),("STR",False),("PER",True)],
    # R9 = Barcelona — required exact top-10 + ANT DNF
    9: [("LEC",False),("RUS",False),("HAM",False),("NOR",False),("PIA",False),("VER",False),("HAD",False),("GAS",False),("LAW",False),("LIN",False),
        ("COL",False),("BEA",False),("SAI",False),("ALB",False),("HUL",False),("BOR",False),("OCO",False),("ALO",False),("BOT",False),("STR",False),("PER",False),("ANT",True)],   # ANT DNF #2
}

# ── Grid positions (starting grid) ───────────────────────────────────────────
GRID_ORDERS = {
    1: ["ANT","RUS","LEC","HAM","NOR","PIA","VER","HAD","GAS","COL","LAW","LIN","BEA","SAI","ALB","HUL","BOR","OCO","ALO","BOT","STR","PER"],
    2: ["ANT","RUS","LEC","HAM","NOR","VER","PIA","HAD","LAW","GAS","COL","BEA","SAI","ALB","LIN","HUL","BOR","OCO","ALO","BOT","STR","PER"],
    3: ["ANT","LEC","RUS","HAM","NOR","PIA","VER","HAD","GAS","LAW","COL","LIN","BEA","SAI","ALB","HUL","BOR","OCO","ALO","BOT","STR","PER"],
    4: ["ANT","HAM","LEC","RUS","NOR","VER","PIA","HAD","GAS","LAW","COL","LIN","BEA","SAI","ALB","HUL","BOR","OCO","ALO","BOT","STR","PER"],
    5: ["RUS","LEC","HAM","ANT","VER","NOR","PIA","GAS","HAD","COL","LAW","LIN","BEA","SAI","ALB","HUL","BOR","OCO","ALO","BOT","STR","PER"],
    6: ["ANT","RUS","LEC","HAM","NOR","VER","PIA","HAD","GAS","BEA","COL","LIN","LAW","SAI","ALB","HUL","BOR","OCO","ALO","BOT","STR","PER"],
    7: ["ANT","RUS","LEC","HAM","NOR","PIA","VER","HAD","GAS","LAW","COL","LIN","BEA","SAI","ALB","HUL","BOR","OCO","ALO","BOT","STR","PER"],
    8: ["RUS","ANT","LEC","HAM","NOR","PIA","VER","HAD","GAS","COL","LIN","LAW","BEA","SAI","ALB","HUL","BOR","OCO","ALO","BOT","STR","PER"],
    9: ["ANT","LEC","RUS","HAM","NOR","PIA","VER","HAD","GAS","LAW","LIN","COL","BEA","SAI","ALB","HUL","BOR","OCO","ALO","BOT","STR","PER"],  # ANT on pole
}

# ── Tyre strategies ───────────────────────────────────────────────────────────
STRATS = ["S-M", "S-M-H", "M-H", "S-H", "M-M"]

def gen_lap_times(base, laps, jitter=0.6):
    """Generate realistic lap times with tyre deg and scatter."""
    times = []
    deg = 0.0
    for i in range(laps):
        if i < 3:
            noise = random.gauss(0, 0.3)
        else:
            noise = random.gauss(0, jitter * 0.5)
            deg += 0.008
        # Occasional slow lap (traffic / yellow flag)
        if random.random() < 0.03:
            noise += random.uniform(2, 8)
        times.append(round(base + deg + noise, 3))
    return times

def gen_telemetry(n_points=100):
    """Generate synthetic telemetry arrays (100 sample points per driver)."""
    distance = [round(i * 46.57, 1) for i in range(n_points)]
    speed, throttle, brake, gear, drs = [], [], [], [], []
    v = 220.0
    for i in range(n_points):
        # crude speed profile: high on straights, low in braking zones
        phase = (i % 20) / 20
        if phase < 0.3:
            v = min(310, v + random.gauss(8, 2))
        elif phase < 0.5:
            v = max(80, v - random.gauss(12, 3))
        else:
            v = min(260, v + random.gauss(5, 2))
        speed.append(round(v, 1))
        thr = 100.0 if v > 200 else max(0, 30 + (v - 80) * 0.7)
        throttle.append(round(min(100, thr + random.gauss(0, 2)), 1))
        brk = 0.0 if v > 200 else max(0, (200 - v) * 0.4)
        brake.append(round(min(100, brk + random.gauss(0, 1)), 1))
        g = max(2, min(8, int(v / 45)))
        gear.append(g)
        drs.append(1 if (v > 240 and phase < 0.3) else 0)
    return {"distance": distance, "speed": speed, "throttle": throttle,
            "brake": brake, "gear": gear, "drs": drs}

# ── Build races ───────────────────────────────────────────────────────────────
def build_race(cal_entry, order):
    rnd  = cal_entry["round"]
    laps = cal_entry["laps"]
    base = cal_entry["base_lap"]
    s1_b, s2_b, s3_b = cal_entry["s1"], cal_entry["s2"], cal_entry["s3"]

    grid_map = {code: pos+1 for pos, code in enumerate(GRID_ORDERS[rnd])}

    results = []
    lap_times = {}
    telemetry = {}

    # Build pace offsets — winner is fastest, each slot a bit slower
    pace = {code: i * 0.07 + random.gauss(0, 0.03) for i, (code, _) in enumerate(order)}

    classified_pos = 0
    for (code, is_dnf) in order:
        info = DRIVER_INFO[code]
        if not is_dnf:
            classified_pos += 1
            pos = classified_pos
            pts = PTS_MAP.get(pos, 0)
            gap = round((pos - 1) * 1.2 + random.gauss(0, 0.3), 3) if pos > 1 else 0.0
            fl  = round(base + pace[code] + random.gauss(0, 0.05), 3)
            s1  = round(s1_b + pace[code] * 0.32 + random.gauss(0, 0.04), 3)
            s2  = round(s2_b + pace[code] * 0.41 + random.gauss(0, 0.04), 3)
            s3  = round(fl - s1 - s2, 3)
            pits = random.randint(1, 2)
            results.append({
                "position": pos, "code": code, "name": info["name"],
                "team": info["team"], "color": info["color"],
                "points": pts, "gap": gap, "fastest_lap": fl,
                "sectors": {"S1": s1, "S2": s2, "S3": s3},
                "pit_stops": pits,
                "tyre_strategy": random.choice(STRATS),
                "grid": grid_map.get(code, 20),
                "status": "Finished",
                "dnf": False,
            })
            lap_times[code] = gen_lap_times(base + pace[code], laps)
        else:
            results.append({
                "position": None, "code": code, "name": info["name"],
                "team": info["team"], "color": info["color"],
                "points": 0, "gap": None, "fastest_lap": None,
                "sectors": None,
                "pit_stops": random.randint(0, 1),
                "tyre_strategy": random.choice(["S", "M"]),
                "grid": grid_map.get(code, 1),
                "status": "DNF",
                "dnf": True,
            })
            lap_times[code] = gen_lap_times(base + pace[code], laps // 2)
        telemetry[code] = gen_telemetry(100)

    return {
        "round": rnd,
        "name": cal_entry["name"],
        "circuit": cal_entry["circuit"],
        "country": cal_entry["country"],
        "date": cal_entry["date"],
        "laps": laps,
        "results": results,
        "lap_times": lap_times,
        "telemetry": telemetry,
    }

# ── Override Barcelona (R9) with exact required results ───────────────────────
def patch_barcelona(race):
    """Ensure R9 top-10 and ANT-DNF match the required spec exactly."""
    required_top10 = [
        ("LEC", 1), ("RUS", 2), ("HAM", 3), ("NOR", 4), ("PIA", 5),
        ("VER", 6), ("HAD", 7), ("GAS", 8), ("LAW", 9), ("LIN", 10),
    ]
    cal = next(c for c in CALENDAR if c["round"] == 9)
    base = cal["base_lap"]
    s1_b, s2_b, s3_b = cal["s1"], cal["s2"], cal["s3"]
    laps = cal["laps"]

    results_by_code = {r["code"]: r for r in race["results"]}

    for code, pos in required_top10:
        r = results_by_code[code]
        r["position"] = pos
        r["points"] = PTS_MAP.get(pos, 0)
        r["gap"] = round((pos - 1) * 1.18 + random.gauss(0, 0.2), 3) if pos > 1 else 0.0
        r["status"] = "Finished"
        r["dnf"] = False
        fl = round(base + (pos - 1) * 0.06 + random.gauss(0, 0.03), 3)
        r["fastest_lap"] = fl
        s1 = round(s1_b + (pos - 1) * 0.02 + random.gauss(0, 0.02), 3)
        s2 = round(s2_b + (pos - 1) * 0.025 + random.gauss(0, 0.02), 3)
        r["sectors"] = {"S1": s1, "S2": s2, "S3": round(fl - s1 - s2, 3)}

    # ANT: DNF from lead, car damage
    ant = results_by_code["ANT"]
    ant["position"] = None
    ant["points"] = 0
    ant["gap"] = None
    ant["fastest_lap"] = None
    ant["sectors"] = None
    ant["status"] = "DNF"
    ant["dnf"] = True
    ant["grid"] = 1   # started from pole

    race["results"] = sorted(
        race["results"],
        key=lambda r: (r["position"] if r["position"] is not None else 999)
    )
    return race

# ── Compute championship standings ────────────────────────────────────────────
def compute_standings(races):
    pts = {d["code"]: 0 for d in DRIVERS}
    for race in races:
        for res in race["results"]:
            pts[res["code"]] = pts.get(res["code"], 0) + res["points"]

    driver_standings = sorted(
        [{"code": c, "name": DRIVER_INFO[c]["name"], "team": DRIVER_INFO[c]["team"],
          "color": DRIVER_INFO[c]["color"], "points": p}
         for c, p in pts.items()],
        key=lambda x: -x["points"]
    )
    # Add championship position
    for i, d in enumerate(driver_standings):
        d["position"] = i + 1

    # Constructors
    team_pts = {}
    for d in driver_standings:
        team_pts[d["team"]] = team_pts.get(d["team"], 0) + d["points"]
    team_colors = {d["team"]: d["color"] for d in DRIVERS}
    constructor_standings = sorted(
        [{"team": t, "color": team_colors.get(t, "#888"), "points": p}
         for t, p in team_pts.items()],
        key=lambda x: -x["points"]
    )
    for i, t in enumerate(constructor_standings):
        t["position"] = i + 1

    return {"drivers": driver_standings, "constructors": constructor_standings}

# ── Main ──────────────────────────────────────────────────────────────────────
races = []
for cal in CALENDAR:
    order = RACE_ORDERS[cal["round"]]
    race  = build_race(cal, order)
    if cal["round"] == 9:
        race = patch_barcelona(race)
    races.append(race)

standings = compute_standings(races)

output = {
    "season": 2026,
    "_note": "Demo data — static fallback committed to repo. Run scripts/fetch_2026_results.py for real data.",
    "points_system": {
        "race":   {str(k): v for k, v in PTS_MAP.items()},
        "sprint": {"1":8,"2":7,"3":6,"4":5,"5":4,"6":3,"7":2,"8":1},
        "fastest_lap_bonus": False,
        "note": "No fastest lap bonus point from 2026",
    },
    "drivers": DRIVERS,
    "races": races,
    "standings": standings,
    "season_info": {
        "total_rounds": 22,
        "completed_rounds": 9,
        "remaining_rounds": 13,
        "leader": standings["drivers"][0]["code"],
        "leader_points": standings["drivers"][0]["points"],
    },
}

out_path = ROOT / "f1_data_2026_demo.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"[OK] Written {out_path}")
print("\nChampionship standings after R9:")
for d in standings["drivers"][:10]:
    print(f"  P{d['position']:>2} {d['code']}  {d['points']} pts")

print("\nBarcelona (R9) results:")
r9 = races[8]
for res in r9["results"]:
    pos = res["position"] if res["position"] else "DNF"
    print(f"  {str(pos):>3}  {res['code']}  status={res['status']}")

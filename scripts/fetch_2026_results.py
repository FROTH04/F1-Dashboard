"""
fetch_2026_results.py — Download real 2026 F1 race results via FastF1.

Usage:
    python scripts/fetch_2026_results.py

Saves results to f1_data_2026.json (rounds that could be fetched).
Falls back gracefully on any network / FastF1 error.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache"
OUT_FILE  = ROOT / "f1_data_2026.json"
DEMO_FILE = ROOT / "f1_data_2026_demo.json"
YEAR = 2026
ROUNDS = range(1, 10)   # official rounds 1-9 (some may be cancelled → skip)

POINTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

# ── Driver / team colour lookup (used to fill missing FastF1 metadata) ────────
DRIVER_COLORS = {
    "ANT": "#27F4D2", "RUS": "#27F4D2",
    "HAM": "#E8002D", "LEC": "#E8002D",
    "NOR": "#FF8000", "PIA": "#FF8000",
    "VER": "#3671C6", "HAD": "#3671C6",
    "GAS": "#FF69B4", "COL": "#FF69B4",
    "LAW": "#6692FF", "LIN": "#6692FF",
    "BEA": "#B6BABD", "OCO": "#B6BABD",
    "BOR": "#F50000", "HUL": "#F50000",
    "SAI": "#64C4FF", "ALB": "#64C4FF",
    "BOT": "#6B9B37", "PER": "#6B9B37",
    "ALO": "#358C75", "STR": "#358C75",
}


def fetch_round(ff1, year: int, rnd: int) -> dict | None:
    """Fetch a single race session. Returns structured dict or None on failure."""
    try:
        sess = ff1.get_session(year, rnd, "R")
        sess.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"[WARNING] Could not fetch round {rnd} — skipping ({type(e).__name__})")
        return None

    if sess.results is None or len(sess.results) == 0:
        print(f"[WARNING] Could not fetch round {rnd} — skipping (no results)")
        return None

    import pandas as pd
    import numpy as np

    results = []
    for _, row in sess.results.iterrows():
        code = str(row.get("Abbreviation", "")).strip().upper()
        if not code:
            continue

        raw_pos = row.get("Position", None)
        try:
            pos = int(raw_pos) if raw_pos is not None and not pd.isna(raw_pos) else None
        except (ValueError, TypeError):
            pos = None

        status = str(row.get("Status", "")).strip()
        is_dnf = pos is None or status.lower() not in ("finished", "+1 lap", "+2 laps",
                                                         "+3 laps", "1 lap", "2 laps")

        # Fastest lap
        fl_raw = row.get("FastestLapTime", None)
        fl_sec = None
        if fl_raw is not None:
            try:
                if hasattr(fl_raw, "total_seconds"):
                    fl_sec = round(fl_raw.total_seconds(), 3)
                elif not pd.isna(fl_raw):
                    fl_sec = round(float(fl_raw), 3)
            except Exception:
                pass

        # Grid position
        grid_raw = row.get("GridPosition", None)
        try:
            grid = int(grid_raw) if grid_raw is not None and not pd.isna(grid_raw) else None
        except (ValueError, TypeError):
            grid = None

        # Pit stops — count stop events from laps data
        pit_stops = 0
        try:
            drv_laps = sess.laps.pick_drivers(code)
            if drv_laps is not None and len(drv_laps) > 0:
                pit_stops = int(drv_laps["PitOutTime"].notna().sum())
        except Exception:
            pass

        team = str(row.get("TeamName", "")).strip() or str(row.get("Constructor", "")).strip()
        name = str(row.get("FullName", "")).strip() or str(row.get("BroadcastName", "")).strip()
        pts  = POINTS_MAP.get(pos, 0) if pos and not is_dnf else 0

        results.append({
            "position":       pos,
            "code":           code,
            "name":           name,
            "team":           team,
            "color":          DRIVER_COLORS.get(code, "#888888"),
            "points":         pts,
            "gap":            None,
            "fastest_lap":    fl_sec,
            "sectors":        None,
            "pit_stops":      pit_stops,
            "tyre_strategy":  None,
            "grid":           grid,
            "status":         status if status else ("DNF" if is_dnf else "Finished"),
            "dnf":            is_dnf,
        })

    if not results:
        print(f"[WARNING] Could not fetch round {rnd} — skipping (empty result set)")
        return None

    # Basic lap times (winner's laps as proxy)
    lap_times: dict = {}
    try:
        for code in [r["code"] for r in results if not r["dnf"]][:5]:
            drv_laps = sess.laps.pick_drivers(code)
            times = []
            for _, lap in drv_laps.iterrows():
                lt = lap.get("LapTime", None)
                if lt is not None and not (hasattr(lt, "__class__") and lt.__class__.__name__ == "NaTType"):
                    try:
                        times.append(round(lt.total_seconds(), 3))
                    except Exception:
                        pass
            if times:
                lap_times[code] = times
    except Exception:
        pass

    ev = sess.event
    return {
        "round":     int(rnd),
        "name":      str(ev.get("EventName", f"Round {rnd}")),
        "circuit":   str(ev.get("Location", "")),
        "country":   str(ev.get("Country", "")),
        "date":      str(ev.get("EventDate", ""))[:10],
        "laps":      int(max((r.get("laps", 0) for r in []), default=0)) if False else None,
        "results":   sorted(results, key=lambda r: (r["position"] or 999)),
        "lap_times": lap_times,
        "telemetry": {},
    }


def load_demo_drivers() -> list:
    """Pull driver list from demo file if available."""
    if DEMO_FILE.exists():
        with open(DEMO_FILE, encoding="utf-8") as f:
            return json.load(f).get("drivers", [])
    return []


def compute_standings(races: list, drivers: list) -> dict:
    pts: dict[str, int] = {}
    for race in races:
        for res in race["results"]:
            c = res["code"]
            pts[c] = pts.get(c, 0) + res.get("points", 0)

    driver_info = {d["code"]: d for d in drivers}
    driver_standings = sorted(
        [{"code": c, "name": driver_info.get(c, {}).get("name", c),
          "team": driver_info.get(c, {}).get("team", ""),
          "color": driver_info.get(c, {}).get("color", "#888"),
          "points": p, "position": 0}
         for c, p in pts.items()],
        key=lambda x: -x["points"],
    )
    for i, d in enumerate(driver_standings):
        d["position"] = i + 1

    team_pts: dict[str, int] = {}
    for d in driver_standings:
        team_pts[d["team"]] = team_pts.get(d["team"], 0) + d["points"]
    team_colors = {d["team"]: d["color"] for d in drivers}
    constructor_standings = sorted(
        [{"team": t, "color": team_colors.get(t, "#888"), "points": p, "position": 0}
         for t, p in team_pts.items()],
        key=lambda x: -x["points"],
    )
    for i, t in enumerate(constructor_standings):
        t["position"] = i + 1

    return {"drivers": driver_standings, "constructors": constructor_standings}


def main():
    try:
        import fastf1
    except ImportError:
        print("[WARNING] FastF1 is not installed — cannot fetch real data.")
        print("[INFO] No live data available. Demo data (f1_data_2026_demo.json) "
              "will be used automatically. This is expected in offline environments.")
        return

    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        fastf1.set_log_level("WARNING")
    except Exception:
        pass

    fetched_races: list[dict] = []

    for rnd in ROUNDS:
        result = fetch_round(fastf1, YEAR, rnd)
        if result is not None:
            fetched_races.append(result)
            print(f"[OK] Fetched round {rnd}: {result['name']}")

    if not fetched_races:
        print("[INFO] No live data available. Demo data (f1_data_2026_demo.json) "
              "will be used automatically. This is expected in offline environments.")
        return

    drivers = load_demo_drivers()
    standings = compute_standings(fetched_races, drivers)

    output = {
        "season": YEAR,
        "points_system": {
            "race":              {str(k): v for k, v in POINTS_MAP.items()},
            "fastest_lap_bonus": False,
            "note":              "No fastest lap bonus point from 2026",
        },
        "drivers": drivers,
        "races":   fetched_races,
        "standings": standings,
        "season_info": {
            "total_rounds":     22,
            "completed_rounds": len(fetched_races),
            "remaining_rounds": 22 - len(fetched_races),
            "leader":       standings["drivers"][0]["code"] if standings["drivers"] else "",
            "leader_points":standings["drivers"][0]["points"] if standings["drivers"] else 0,
        },
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved {len(fetched_races)} rounds to {OUT_FILE.name}")


if __name__ == "__main__":
    main()

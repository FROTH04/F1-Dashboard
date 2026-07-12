"""
ml_models.py — F1 Race Outcome Prediction Models
==================================================
Trains three models on historical F1 race data:
  1. Random Forest Regressor  → predicted finishing position
  2. XGBoost Regressor        → predicted finishing position (ensemble comparison)
  3. LSTM (PyTorch or NumPy fallback) → temporal trend model over the season

Feature engineering uses race-level aggregates from FastF1 / demo JSON.
Strict chronological train/test split (no data leakage).

Usage:
    python ml_models.py              # trains & evaluates, saves to models/
    python ml_models.py --demo       # forces demo JSON data (no FastF1 needed)
"""

import os, sys, json, argparse, warnings, logging
# Windows terminals default to cp1252 and cannot render some unicode symbols;
# force UTF-8 so log output is identical on Linux and Windows.
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import pickle
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEASON = 2026

# Public API — championship_simulator.py, shap_analysis.py and f1_dashboard.py
# import from here; keep this list in sync when adding shared symbols.
__all__ = [
    "SEASON",
    "FEATURE_COLS",
    "TEAM_STRENGTH",
    "DRIVER_EXP_2026",
    "POINTS_MAP",
    "TRACK_PROFILES",
    "build_team_dna",
    "get_team_dna",
    "get_effective_team_dna",
    "get_track_profile",
    "load_context_updates",
    "load_demo_data",
    "load_fastf1_data",
    "build_feature_matrix",
    "build_inference_matrix",
    "build_lstm_sequences",
    "compute_reliability",
    "compute_pace_adjusted_form",
    "compute_momentum",
    "is_dnf_result",
    "LSTMNumpy",
]

# ── Team strength mapping (mid-field vs top teams) — 2026 season ─────────────
# Mercedes leads in 2026 with ANT/RUS 1-2; Ferrari strong; McLaren 3rd
TEAM_STRENGTH = {
    "Mercedes": 10, "Ferrari": 9, "McLaren": 8, "Red Bull": 7,
    "Alpine": 6, "Racing Bulls": 5, "Haas": 4, "Williams": 4,
    "Audi": 4, "Aston Martin": 3, "Cadillac": 3,
    # Legacy names for compatibility
    "Red Bull Racing": 7, "Kick Sauber": 3, "Sauber": 3,
}

DRIVER_EXPERIENCE = {
    # 2026 real driver grid
    "ANT": 2,  "RUS": 7,  "HAM": 20, "LEC": 7,
    "NOR": 6,  "PIA": 3,  "VER": 9,  "HAD": 1,
    "GAS": 10, "COL": 2,  "LAW": 3,  "LIN": 1,
    "BEA": 2,  "OCO": 9,  "BOR": 1,  "HUL": 12,
    "SAI": 10, "ALB": 7,  "BOT": 14, "PER": 10,
    "ALO": 24, "STR": 7,
}
# Public alias used by championship_simulator.py and f1_dashboard.py
DRIVER_EXP_2026 = DRIVER_EXPERIENCE

# Standard 2026 points system (no fastest-lap bonus point)
POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


# Module-level cache populated by build_team_dna(); used by get_team_dna().
_TEAM_DNA_CACHE: dict = {}

# ── Per-track sensitivity profiles ───────────────────────────────────────────
# power/downforce ∈ [0,1] = how much each matters here.
# type: 0=power, 1=balanced, 2=downforce, 3=street.  speed = avg km/h (context).
TRACK_PROFILES = {
    "Australian Grand Prix":  {"power": 0.65, "downforce": 0.70, "type": 1, "speed": 210},
    "Chinese Grand Prix":     {"power": 0.70, "downforce": 0.60, "type": 1, "speed": 205},
    "Japanese Grand Prix":    {"power": 0.60, "downforce": 0.80, "type": 2, "speed": 200},
    "Miami Grand Prix":       {"power": 0.65, "downforce": 0.65, "type": 1, "speed": 215},
    "Canadian Grand Prix":    {"power": 0.80, "downforce": 0.50, "type": 0, "speed": 205},
    "Monaco Grand Prix":      {"power": 0.25, "downforce": 0.95, "type": 3, "speed": 155},
    "Barcelona Grand Prix":   {"power": 0.60, "downforce": 0.80, "type": 2, "speed": 200},
    "Austrian Grand Prix":    {"power": 0.70, "downforce": 0.65, "type": 0, "speed": 230},
    "British Grand Prix":     {"power": 0.72, "downforce": 0.75, "type": 1, "speed": 235},
    "Belgian GP":             {"power": 0.88, "downforce": 0.50, "type": 0, "speed": 240},
    "Hungarian GP":           {"power": 0.30, "downforce": 0.92, "type": 2, "speed": 185},
    "Dutch GP":               {"power": 0.45, "downforce": 0.88, "type": 2, "speed": 195},
    "Italian GP":             {"power": 0.95, "downforce": 0.20, "type": 0, "speed": 255},
    "Spanish GP":             {"power": 0.60, "downforce": 0.75, "type": 3, "speed": 200},
    "Azerbaijan GP":          {"power": 0.85, "downforce": 0.45, "type": 3, "speed": 215},
    "Singapore GP":           {"power": 0.35, "downforce": 0.90, "type": 3, "speed": 170},
    "US GP":                  {"power": 0.68, "downforce": 0.72, "type": 1, "speed": 210},
    "Mexico City GP":         {"power": 0.78, "downforce": 0.55, "type": 0, "speed": 215},
    "São Paulo GP":           {"power": 0.65, "downforce": 0.70, "type": 1, "speed": 205},
    "Las Vegas GP":           {"power": 0.90, "downforce": 0.35, "type": 3, "speed": 245},
    "Qatar GP":               {"power": 0.55, "downforce": 0.80, "type": 2, "speed": 215},
    "Abu Dhabi GP":           {"power": 0.65, "downforce": 0.68, "type": 1, "speed": 210},
}
BALANCED_PROFILE = {"power": 0.65, "downforce": 0.68, "type": 1, "speed": 210}

# Map assorted race-name spellings (calendar "GP" vs data "Grand Prix") → profile keys
_TRACK_NAME_ALIASES = {
    "Australian GP": "Australian Grand Prix",
    "Chinese GP": "Chinese Grand Prix",
    "Japanese GP": "Japanese Grand Prix",
    "Miami GP": "Miami Grand Prix",
    "Canadian GP": "Canadian Grand Prix",
    "Monaco GP": "Monaco Grand Prix",
    "Barcelona GP": "Barcelona Grand Prix",
    "Spanish Grand Prix": "Spanish GP",
    "Austrian GP": "Austrian Grand Prix",
    "British GP": "British Grand Prix",
    "Belgian Grand Prix": "Belgian GP",
    "Hungarian Grand Prix": "Hungarian GP",
    "Dutch Grand Prix": "Dutch GP",
    "Italian Grand Prix": "Italian GP",
    "Azerbaijan Grand Prix": "Azerbaijan GP",
    "Singapore Grand Prix": "Singapore GP",
    "United States GP": "US GP",
    "US Grand Prix": "US GP",
    "Mexico City Grand Prix": "Mexico City GP",
    "São Paulo Grand Prix": "São Paulo GP",
    "Sao Paulo GP": "São Paulo GP",
    "Las Vegas Grand Prix": "Las Vegas GP",
    "Qatar Grand Prix": "Qatar GP",
    "Abu Dhabi Grand Prix": "Abu Dhabi GP",
}


def get_track_profile(race_name):
    """Resolve a race name to its track profile, tolerating naming variants."""
    if race_name in TRACK_PROFILES:
        return TRACK_PROFILES[race_name]
    alias = _TRACK_NAME_ALIASES.get(race_name)
    if alias and alias in TRACK_PROFILES:
        return TRACK_PROFILES[alias]
    return BALANCED_PROFILE


def load_context_updates():
    """Load context_updates.json if present, else return empty context."""
    ctx_file = ROOT / "context_updates.json"
    if ctx_file.exists():
        try:
            with open(ctx_file, encoding="utf-8") as f:
                ctx = json.load(f)
            ctx.setdefault("team_updates", [])
            ctx.setdefault("driver_updates", [])
            return ctx
        except Exception as e:
            log.warning(f"  Warning: could not read {ctx_file.name}: {e}")
    return {}


def build_team_dna(data, context_updates=None):
    """
    Derive team DNA (engine / chassis ratings 0-10) from actual race data.

    pace_delta = fastest_lap - session_fastest_lap  (gap to the session's
    fastest lap; >= 0, smaller = faster).  Ratings:
        engine_rating  = clamp(10.0 - avg_pace_delta_on_power_tracks    * 8.0)
        chassis_rating = clamp(10.0 - avg_pace_delta_on_downforce_tracks * 8.0)
    clamped to [3.0, 10.0].  Falls back to calibrated 2026 constants when
    fewer than 2 power-track races exist in the loaded data.  Context updates
    (context_updates.json) are applied afterwards and re-clamped.
    Populates _TEAM_DNA_CACHE so get_team_dna() / get_effective_team_dna() use it.
    """
    global _TEAM_DNA_CACHE
    context_updates = context_updates or {}

    _FALLBACK = {
        "Mercedes":     {"engine": 8.5, "chassis": 7.8},
        "Ferrari":      {"engine": 7.2, "chassis": 8.8},
        "McLaren":      {"engine": 8.2, "chassis": 8.0},
        "Red Bull":     {"engine": 7.8, "chassis": 7.5},
        "Alpine":       {"engine": 7.5, "chassis": 6.2},
        "Racing Bulls": {"engine": 7.8, "chassis": 5.8},
        "Haas":         {"engine": 7.2, "chassis": 5.2},
        "Williams":     {"engine": 8.2, "chassis": 5.0},
        "Audi":         {"engine": 5.8, "chassis": 5.5},
        "Aston Martin": {"engine": 7.2, "chassis": 5.0},
        "Cadillac":     {"engine": 6.2, "chassis": 4.5},
    }
    _DNA_DEFAULT = {"engine": 6.0, "chassis": 6.0}
    _POWER_KW    = {"canada", "spa", "monza", "baku", "las vegas", "jeddah", "austria"}
    _DOWNFORCE_KW = {"monaco", "hungary", "singapore", "zandvoort", "japan", "barcelona"}

    team_power_deltas: dict     = {}
    team_downforce_deltas: dict = {}
    power_race_count = 0

    for race in sorted(data.get("races", []), key=lambda r: r["round"]):
        profile   = get_track_profile(race.get("name", ""))
        name_low  = race.get("name", "").lower()
        is_power     = profile["power"]     > 0.75 or any(kw in name_low for kw in _POWER_KW)
        is_downforce = profile["downforce"] > 0.75 or any(kw in name_low for kw in _DOWNFORCE_KW)
        if not (is_power or is_downforce):
            continue

        results = race.get("results", [])
        laps = [r["fastest_lap"] for r in results if (r.get("fastest_lap") or 0) > 60]
        # Reference is the SESSION FASTEST lap, so every delta is >= 0 and the
        # quickest team scores near 10.0 (a field median would give half the
        # grid negative deltas and clamp all top teams to exactly 10.0).
        session_fastest = float(min(laps)) if laps else 90.0

        if is_power:
            power_race_count += 1

        for r in results:
            team = r.get("team", "")
            fl   = r.get("fastest_lap") or 0.0
            if not team or fl <= 60:
                continue
            delta = fl - session_fastest
            if is_power:
                team_power_deltas.setdefault(team, []).append(delta)
            if is_downforce:
                team_downforce_deltas.setdefault(team, []).append(delta)

    # --- fallback path ---
    if power_race_count < 2:
        print("[WARNING] Insufficient track data for Team DNA derivation - "
              "using calibrated 2026 fallbacks.", flush=True)
        team_dna = {t: dict(v) for t, v in _FALLBACK.items()}
        derived_from_data: set = set()
    else:
        all_teams = set(team_power_deltas) | set(team_downforce_deltas) | set(_FALLBACK)
        team_dna = {}
        derived_from_data = set()

        for team in all_teams:
            p_d = team_power_deltas.get(team, [])
            d_d = team_downforce_deltas.get(team, [])
            fb  = _FALLBACK.get(team, _DNA_DEFAULT)

            engine  = (max(3.0, min(10.0, 10.0 - float(np.mean(p_d)) * 8))
                       if len(p_d) >= 2 else fb["engine"])
            chassis = (max(3.0, min(10.0, 10.0 - float(np.mean(d_d)) * 8))
                       if len(d_d) >= 2 else fb["chassis"])

            team_dna[team] = {"engine": engine, "chassis": chassis}
            if len(p_d) >= 2 or len(d_d) >= 2:
                derived_from_data.add(team)

    # --- legacy aliases ---
    if "Red Bull" in team_dna:
        team_dna["Red Bull Racing"] = dict(team_dna["Red Bull"])
    if "Audi" in team_dna:
        team_dna["Sauber"]      = dict(team_dna["Audi"])
        team_dna["Kick Sauber"] = dict(team_dna["Audi"])

    # Cache the PRE-context base ratings.  get_effective_team_dna() applies
    # news-context deltas per round (respecting from_round); baking them into
    # the cache as well would apply every update twice.
    _TEAM_DNA_CACHE = {t: dict(v) for t, v in team_dna.items()}

    # --- context overrides (returned/printed values reflect current news) ---
    for upd in context_updates.get("team_updates", []):
        team = upd.get("team", "")
        if not team or team not in team_dna:
            continue
        if upd.get("type") in ("engine_upgrade", "engine_delta"):
            team_dna[team]["engine"] = max(3.0, min(10.0,
                team_dna[team]["engine"] + float(upd.get("engine_delta", 0) or 0)))
        if upd.get("type") in ("chassis_upgrade", "chassis_delta"):
            team_dna[team]["chassis"] = max(3.0, min(10.0,
                team_dna[team]["chassis"] + float(upd.get("chassis_delta", 0) or 0)))

    # --- print ---
    _ALIASES = {"Red Bull Racing", "Sauber", "Kick Sauber"}
    print("\n  Team DNA derived from race data:", flush=True)
    for team, dna in sorted(team_dna.items()):
        if team in _ALIASES:
            continue
        src = "data" if team in derived_from_data else "fallback"
        print(f"    {team:<20s} engine={dna['engine']:.1f}  chassis={dna['chassis']:.1f}  [{src}]",
              flush=True)

    return team_dna


def get_team_dna(team, team_dna=None):
    """Team DNA with graceful fallback for unknown teams."""
    source = team_dna if team_dna is not None else _TEAM_DNA_CACHE
    return source.get(team, {"engine": 6.0, "chassis": 6.0})


def get_effective_team_dna(team, round_num, context=None):
    """
    Team DNA adjusted by any news-context team updates in force at `round_num`
    (e.g. Ferrari's engine upgrade from round 12).  Never mutates TEAM_DNA.
    """
    base = get_team_dna(team)
    dna = {"engine": base["engine"], "chassis": base["chassis"]}
    if not context:
        return dna
    for upd in context.get("team_updates", []):
        if upd.get("team") == team and (upd.get("from_round") or 10 ** 9) <= round_num:
            if upd.get("engine_delta") is not None:
                dna["engine"] = min(10.0, dna["engine"] + float(upd["engine_delta"]))
            if upd.get("chassis_delta") is not None:
                dna["chassis"] = min(10.0, dna["chassis"] + float(upd["chassis_delta"]))
    return dna


# Weights for last-3-race rolling stats (oldest → most recent).
_ROLL_WEIGHTS = [0.20, 0.30, 0.50]


def _weighted_recent(values, weights=_ROLL_WEIGHTS):
    """Weighted mean of the last len(values) items, aligning to most-recent weight."""
    if not values:
        return None
    v = list(values)[-len(weights):]
    w = weights[-len(v):]
    tot = sum(w)
    return sum(wi * vi for wi, vi in zip(w, v)) / tot if tot else float(np.mean(v))


def compute_reliability(dnf_flags):
    """
    Rolling DNF stats from a driver's chronological DNF-flag history.
    Returns (rolling_dnf_rate, consecutive_dnf_flag, reliability_concern_score).
    """
    last3 = [1 if d else 0 for d in dnf_flags[-3:]]
    padded = [0] * (3 - len(last3)) + last3          # left-pad to length 3
    rate = sum(w * d for w, d in zip(_ROLL_WEIGHTS, padded))
    if sum(padded) >= 2:
        rate *= 2.5
    rate = min(0.8, rate)

    last2 = [1 if d else 0 for d in dnf_flags[-2:]]
    consecutive = 1 if len(last2) == 2 and sum(last2) == 2 else 0

    concern = 0.4 * rate + 0.6 * consecutive
    return rate, consecutive, concern


def compute_pace_adjusted_form(recent):
    """
    Pace-adjusted finishing form from the last 3 races.
    `recent` = list of dicts oldest→newest with keys: position, dnf, pace_delta.
    A DNF with faster-than-median pace uses pace (not the retirement position).
    """
    if not recent:
        return 11.0
    adj = []
    for r in recent[-3:]:
        if r["dnf"] and r["pace_delta"] < 0:
            adj.append(11.0 + r["pace_delta"] * 8.0)   # faster pace → better equiv pos
        else:
            adj.append(float(r["position"] or 20))
    return _weighted_recent(adj)


def compute_momentum(pace_deltas):
    """
    Momentum from the trend of the last 3 pace deltas (seconds vs field median).
    Improving (falling delta = negative slope) → positive momentum in [-1, 1].
    """
    pd = list(pace_deltas[-3:])
    if len(pd) < 2:
        return 0.0
    slope = float(np.polyfit(np.arange(len(pd)), pd, 1)[0])
    return float(np.tanh(-slope * 5.0))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_race_data(base_dir: Path = ROOT) -> dict:
    """Load race data — real if available, demo as fallback."""
    real = base_dir / "f1_data_2026.json"
    demo = base_dir / "f1_data_2026_demo.json"
    if real.exists():
        with open(real, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OK] Using real race data ({len(data['races'])} rounds)")
        return data
    else:
        with open(demo, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Using demo race data — run scripts/fetch_2026_results.py for real data")
        return data


def load_demo_data() -> dict:
    """Backward-compatible alias for load_race_data()."""
    return load_race_data(ROOT)


def load_fastf1_data(year=2026):
    """Try to load real FastF1 data; fall back to demo on any error."""
    try:
        import fastf1
        cache_dir = ROOT / "f1_cache"
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        fastf1.set_log_level("WARNING")

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        today = pd.Timestamp.now().tz_localize(None)
        races_raw = []

        for _, ev in schedule.iterrows():
            ev_date = pd.Timestamp(ev["EventDate"])
            if ev_date.tzinfo:
                ev_date = ev_date.tz_localize(None)
            if ev_date >= today:
                continue
            try:
                sess = fastf1.get_session(year, int(ev["RoundNumber"]), "R")
                sess.load(laps=True, telemetry=False, weather=False, messages=False)
                if len(sess.results) == 0:
                    continue

                PTS = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
                results = []
                for _, row in sess.results.iterrows():
                    code = str(row.get("Abbreviation", ""))
                    pos = int(row.get("Position", 99)) if not pd.isna(row.get("Position", np.nan)) else 99
                    dl = sess.laps.pick_driver(code) if len(sess.laps) > 0 else pd.DataFrame()
                    fl_t = 90.0
                    if len(dl) > 0:
                        try:
                            fl = dl.pick_fastest()
                            t = fl["LapTime"]
                            fl_t = t.total_seconds() if hasattr(t, "total_seconds") else float(t)
                        except Exception:
                            pass
                    results.append({
                        "position": pos, "code": code,
                        "team": str(row.get("TeamName", "")),
                        "fastest_lap": fl_t,
                        "points": PTS.get(pos, 0),
                        "gap": 0.0,
                    })

                races_raw.append({
                    "round": int(ev["RoundNumber"]),
                    "name": ev["EventName"],
                    "results": sorted(results, key=lambda x: x["position"]),
                })
            except Exception as e:
                log.debug(f"  Skipping R{ev['RoundNumber']}: {e}")

        if len(races_raw) >= 3:
            log.info(f"  FastF1: loaded {len(races_raw)} races for {year}")
            return {"races": races_raw}
    except Exception as e:
        log.info(f"  FastF1 not available ({e}), using demo data.")

    return load_demo_data()


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "round",
    "track_power_sensitivity",
    "track_downforce_sensitivity",
    "track_type_encoded",
    "engine_rating",
    "chassis_rating",
    "track_performance_score",
    "pace_adjusted_form",
    "momentum_score",
    "pace_delta_s1",
    "pace_delta_s2",
    "pace_delta_s3",
    "rolling_dnf_rate",
    "consecutive_dnf_flag",
    "reliability_concern_score",
    "cumulative_points",
    "driver_experience",
    "championship_position",
]


def is_dnf_result(result, demo=True):
    """
    Derive DNF status FROM DATA (never hardcoded).
      - Demo mode: explicit dnf flag, null position, or finishing position >= 20.
      - FastF1 mode: status field indicates non-finish AND position >= 18.
    """
    pos = result.get("position")
    if demo:
        return (
            bool(result.get("dnf", False))
            or str(result.get("status", "")).upper() == "DNF"
            or pos is None
            or (isinstance(pos, int) and pos >= 20)
        )
    status = str(result.get("status", "Finished"))
    finished = ("Finished", "+1 Lap", "+2 Laps", "+3 Laps")
    return (status not in finished) and (pos is None or (isinstance(pos, int) and pos >= 18))


def synth_sector_deltas(engine_rating, chassis_rating, track_score, rng=None):
    """Synthesise per-sector pace deltas from team DNA (demo / inference mode)."""
    rng = rng or np.random
    s1 = -0.08 * (engine_rating - 7.0) / 3.0 + rng.normal(0, 0.03)
    s2 = -0.08 * (chassis_rating - 7.0) / 3.0 + rng.normal(0, 0.03)
    s3 = -0.05 * track_score / 10.0 + rng.normal(0, 0.02)
    return float(s1), float(s2), float(s3)


def track_performance_score(engine, chassis, power_s, downforce_s):
    """Weighted blend of engine/chassis by how much the track rewards each."""
    denom = power_s + downforce_s
    if denom <= 0:
        return (engine + chassis) / 2.0
    return (engine * power_s + chassis * downforce_s) / denom


def build_feature_matrix(data, context=None, demo=True, rng=None):
    """
    Build a feature DataFrame from race results.  One row = one driver in one race.

    All 18 features (see FEATURE_COLS) are computable strictly from prior rounds
    plus the current track/team profile.  Track profile comes from TRACK_PROFILES,
    team strength from build_team_dna() (derived from race pace data, with
    calibrated fallbacks).  DNF history is DERIVED from data, not hardcoded.

    Targets kept for training:  position (regression), dnf (classification).
    """
    rng = rng or np.random
    context = context or {}
    records = []
    races = sorted(data["races"], key=lambda r: r["round"])

    # Derive team DNA from race data (sets _TEAM_DNA_CACHE for get_effective_team_dna)
    build_team_dna(data, load_context_updates())

    # Chronological per-driver history of {position, dnf, pace_delta, points}
    driver_history: dict[str, list[dict]] = {}
    cum_points: dict[str, int] = {}

    for race in races:
        round_num = race["round"]
        results = race.get("results", [])
        if not results:
            continue

        track = get_track_profile(race.get("name", ""))
        power_s, downforce_s = track["power"], track["downforce"]

        # Field-median fastest lap for pace deltas
        laps = [r["fastest_lap"] for r in results if (r.get("fastest_lap") or 0) > 60]
        field_median_lap = float(np.median(laps)) if laps else 90.0

        # Championship position at this round: rank by points BEFORE this race,
        # ties broken alphabetically by driver code.
        pre_race_points = {r["code"]: cum_points.get(r["code"], 0) for r in results}
        ranked = sorted(pre_race_points.items(), key=lambda kv: (-kv[1], kv[0]))
        champ_pos = {code: i + 1 for i, (code, _) in enumerate(ranked)}

        pending_updates = []
        for r in results:
            code = r["code"]
            pos = r["position"]
            team = r.get("team", "")
            dna = get_effective_team_dna(team, round_num, context)
            engine, chassis = dna["engine"], dna["chassis"]

            score = track_performance_score(engine, chassis, power_s, downforce_s)
            s1, s2, s3 = synth_sector_deltas(engine, chassis, score, rng)

            hist = driver_history.get(code, [])
            rate, consec, concern = compute_reliability([h["dnf"] for h in hist])
            pace_form = compute_pace_adjusted_form(hist)
            momentum = compute_momentum([h["pace_delta"] for h in hist])

            records.append({
                "round":                        round_num,
                "driver":                       code,
                "team":                         team,
                "position":                     pos,
                "dnf":                          int(is_dnf_result(r, demo)),
                "track_power_sensitivity":      power_s,
                "track_downforce_sensitivity":  downforce_s,
                "track_type_encoded":           track["type"],
                "engine_rating":                engine,
                "chassis_rating":               chassis,
                "track_performance_score":      score,
                "pace_adjusted_form":           pace_form,
                "momentum_score":               momentum,
                "pace_delta_s1":                s1,
                "pace_delta_s2":                s2,
                "pace_delta_s3":                s3,
                "rolling_dnf_rate":             rate,
                "consecutive_dnf_flag":         consec,
                "reliability_concern_score":    concern,
                "cumulative_points":            cum_points.get(code, 0),
                "driver_experience":            DRIVER_EXP_2026.get(code, 3),
                "championship_position":        champ_pos.get(code, 11),
            })

            pace_delta = (r.get("fastest_lap") or field_median_lap) - field_median_lap
            pending_updates.append((code, {
                "round": round_num,
                "position": pos,
                "dnf": is_dnf_result(r, demo),
                "pace_delta": pace_delta,
                "points": r.get("points", 0),
            }))

        # Commit history AFTER the whole race so within-race features stay pre-race
        for code, entry in pending_updates:
            driver_history.setdefault(code, []).append(entry)
            cum_points[code] = cum_points.get(code, 0) + entry["points"]

    df = pd.DataFrame(records)
    log.info(f"  Feature matrix: {len(df)} rows × {len(df.columns)} columns "
             f"({len(FEATURE_COLS)} model features)")
    return df


def build_inference_matrix(round_num, race_name, standings, base_df,
                           context=None, rng=None):
    """
    Build an inference feature matrix (one row per driver) for an UPCOMING race.

    Rolling driver-state (pace/momentum/reliability) is carried over from each
    driver's most recent completed race in `base_df`; track- and team-dependent
    features are recomputed for the selected round (with context updates applied).

    Returns (X ndarray [n,18], codes list, states dict code→feature-dict).
    """
    rng = rng or np.random
    context = context or {}
    track = get_track_profile(race_name)
    power_s, downforce_s = track["power"], track["downforce"]

    # Championship position from current standings points.
    ranked = sorted(standings.items(), key=lambda kv: (-kv[1]["points"], kv[0]))
    champ_pos = {code: i + 1 for i, (code, _) in enumerate(ranked)}

    codes, rows, states = [], [], {}
    for code, info in standings.items():
        team = info["team"]
        dna = get_effective_team_dna(team, round_num, context)
        engine, chassis = dna["engine"], dna["chassis"]
        score = track_performance_score(engine, chassis, power_s, downforce_s)
        s1, s2, s3 = synth_sector_deltas(engine, chassis, score, rng)

        drv = base_df[base_df["driver"] == code].sort_values("round")
        if len(drv) > 0:
            last = drv.iloc[-1]
            pace_form = float(last["pace_adjusted_form"])
            momentum = float(last["momentum_score"])
            rate = float(last["rolling_dnf_rate"])
            consec = int(last["consecutive_dnf_flag"])
            concern = float(last["reliability_concern_score"])
        else:
            pace_form, momentum, rate, consec, concern = 11.0, 0.0, 0.0, 0, 0.0

        feat = {
            "round":                        round_num,
            "track_power_sensitivity":      power_s,
            "track_downforce_sensitivity":  downforce_s,
            "track_type_encoded":           track["type"],
            "engine_rating":                engine,
            "chassis_rating":               chassis,
            "track_performance_score":      score,
            "pace_adjusted_form":           pace_form,
            "momentum_score":               momentum,
            "pace_delta_s1":                s1,
            "pace_delta_s2":                s2,
            "pace_delta_s3":                s3,
            "rolling_dnf_rate":             rate,
            "consecutive_dnf_flag":         consec,
            "reliability_concern_score":    concern,
            "cumulative_points":            info["points"],
            "driver_experience":            DRIVER_EXP_2026.get(code, 3),
            "championship_position":        champ_pos.get(code, 11),
        }
        codes.append(code)
        rows.append([feat[c] for c in FEATURE_COLS])
        states[code] = feat

    return np.array(rows, dtype=np.float32), codes, states


# ═══════════════════════════════════════════════════════════════════════════════
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_split(df, test_rounds=2):
    """
    Split strictly by race round — last `test_rounds` races are the test set.
    This prevents data leakage (a future race cannot inform past predictions).
    """
    rounds = sorted(int(r) for r in df["round"].unique())
    if len(rounds) <= test_rounds:
        test_rounds = 1
    train_rounds = rounds[:-test_rounds]
    test_rounds_list = rounds[-test_rounds:]

    train = df[df["round"].isin(train_rounds)].copy()
    test  = df[df["round"].isin(test_rounds_list)].copy()
    log.info(f"  Train rounds: {train_rounds}  |  Test rounds: {test_rounds_list}")
    return train, test


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: RANDOM FOREST REGRESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def train_random_forest(train, test):
    """
    RandomForestRegressor for finishing position prediction.
    Returns trained model + evaluation metrics.
    """
    log.info("\n[1/3] Training Random Forest Regressor...")

    X_train = train[FEATURE_COLS].values
    y_train = train["position"].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["position"].values

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    # Position accuracy within ±2 places
    acc2 = np.mean(np.abs(preds - y_test) <= 2)

    log.info(f"  MAE: {mae:.2f} positions  |  Within ±2 places: {acc2*100:.1f}%")

    # Feature importances
    importances = dict(zip(FEATURE_COLS, rf.feature_importances_))
    log.info("  Top features: " + ", ".join(
        f"{k}={v:.3f}" for k, v in sorted(importances.items(), key=lambda x: -x[1])[:4]
    ))

    return rf, {"mae": mae, "acc2": acc2, "importances": importances,
                "predictions": preds.tolist(), "actuals": y_test.tolist()}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: XGBOOST REGRESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(train, test):
    """
    XGBoost Regressor for finishing position prediction.
    Compared against Random Forest as a stronger ensemble baseline.
    """
    log.info("\n[2/3] Training XGBoost Regressor...")

    X_train = train[FEATURE_COLS].values
    y_train = train["position"].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["position"].values

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    acc2 = np.mean(np.abs(preds - y_test) <= 2)

    log.info(f"  MAE: {mae:.2f} positions  |  Within ±2 places: {acc2*100:.1f}%")

    return xgb_model, {"mae": mae, "acc2": acc2,
                       "predictions": preds.tolist(), "actuals": y_test.tolist()}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3a: DNF CLASSIFIER (Gradient Boosting)
# ═══════════════════════════════════════════════════════════════════════════════

def train_dnf_classifier(train, test):
    """
    Gradient Boosting Classifier for DNF / retirement probability.
    Binary: 1 = did not finish (position >= 20), 0 = finished.
    """
    log.info("\n[Bonus] Training DNF Classifier (Gradient Boosting)...")

    X_train = train[FEATURE_COLS].values
    y_train = train["dnf"].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["dnf"].values

    # Only train if we have both classes
    if len(np.unique(y_train)) < 2:
        log.info("  Skipped: no DNF examples in training data.")
        return None, {}

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    log.info(f"  Accuracy: {acc*100:.1f}%")

    return clf, {"accuracy": acc}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3b: LSTM for Temporal Trends
# ═══════════════════════════════════════════════════════════════════════════════

def build_lstm_sequences(df, seq_len=5):
    """
    Build (X, y) sequences for LSTM from per-driver race histories.
    X shape: (samples, seq_len, n_features)
    y shape: (samples,)  — next race position
    """
    sequences, targets, drivers_out = [], [], []
    feature_cols_lstm = list(FEATURE_COLS)   # LSTM consumes the full 18-feature set

    for driver in df["driver"].unique():
        drv_df = df[df["driver"] == driver].sort_values("round")
        if len(drv_df) < seq_len + 1:
            continue
        vals = drv_df[feature_cols_lstm].values
        targets_vals = drv_df["position"].values
        for i in range(len(vals) - seq_len):
            sequences.append(vals[i:i + seq_len])
            targets.append(targets_vals[i + seq_len])
            drivers_out.append(driver)

    if not sequences:
        return None, None, None, feature_cols_lstm

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return X, y, drivers_out, feature_cols_lstm


class LSTMNumpy:
    """
    Minimal single-layer LSTM in pure NumPy — no PyTorch dependency.
    Used as fallback when torch is not installed.

    Architecture:
      input (seq_len, n_features) → LSTM hidden states → last hidden → linear → position
    """

    def __init__(self, input_size, hidden_size=32, output_size=1):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        rs = np.random.RandomState(42)
        scale = 0.1

        # LSTM gate weights  (4 gates: i, f, g, o)
        self.Wx = rs.randn(4 * hidden_size, input_size)  * scale
        self.Wh = rs.randn(4 * hidden_size, hidden_size) * scale
        self.b  = np.zeros(4 * hidden_size)

        # Output linear layer
        self.Wo = rs.randn(output_size, hidden_size) * scale
        self.bo = np.zeros(output_size)

        self.scaler = StandardScaler()
        self.fitted = False

    def _sigmoid(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    def _tanh(self, x):    return np.tanh(np.clip(x, -50, 50))

    def _forward_sequence(self, x):
        """x: (seq_len, input_size) → last hidden state (hidden_size,)"""
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for t in range(x.shape[0]):
            gates = self.Wx @ x[t] + self.Wh @ h + self.b
            H = self.hidden_size
            i = self._sigmoid(gates[0*H:1*H])
            f = self._sigmoid(gates[1*H:2*H])
            g = self._tanh(  gates[2*H:3*H])
            o = self._sigmoid(gates[3*H:4*H])
            c = f * c + i * g
            h = o * self._tanh(c)
        return h

    def _compute_gradients(self, X, y, lr=0.01):
        """Single SGD step (simplified BPTT — last hidden only)."""
        total_loss = 0.0
        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db  = np.zeros_like(self.b)

        for i in range(len(X)):
            h = self._forward_sequence(X[i])
            pred = (self.Wo @ h + self.bo)[0]
            err = pred - y[i]
            total_loss += err ** 2

            # Output layer gradients
            d_out = np.array([err])
            dWo += np.outer(d_out, h)
            dbo += d_out

            # Backprop into LSTM (approximate: only through last step)
            dh = self.Wo.T @ d_out   # (hidden_size,)
            # dWx has shape (4*H, input_size); use only the output-gate rows (3H:4H)
            x_last = X[i, -1]        # (input_size,)
            H = self.hidden_size
            dWx[3*H:4*H] += np.outer(dh, x_last)
            dWh[3*H:4*H] += np.outer(dh, np.zeros(H))
            db[3*H:4*H]  += dh

        n = len(X)
        self.Wo -= lr * dWo / n
        self.bo -= lr * dbo / n
        self.Wx -= lr * np.clip(dWx / n, -1, 1)
        self.Wh -= lr * np.clip(dWh / n, -1, 1)
        self.b  -= lr * np.clip(db / n, -1, 1)

        return total_loss / n

    def fit(self, X, y, epochs=50, lr=0.005):
        """Train LSTM with SGD."""
        # Normalize features
        n, seq, feat = X.shape
        X_flat = X.reshape(-1, feat)
        self.scaler.fit(X_flat)
        X_norm = self.scaler.transform(X_flat).reshape(n, seq, feat)

        # Normalize target
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-6
        y_norm = (y - self.y_mean) / self.y_std

        log.info(f"  Training NumPy LSTM ({epochs} epochs)...")
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n)
            loss = self._compute_gradients(X_norm[idx], y_norm[idx], lr)
            if (epoch + 1) % 10 == 0:
                log.info(f"    Epoch {epoch+1:3d}/{epochs}  loss={loss:.4f}")

        self.fitted = True
        return self

    def predict(self, X):
        n, seq, feat = X.shape
        X_flat = X.reshape(-1, feat)
        X_norm = self.scaler.transform(X_flat).reshape(n, seq, feat)
        preds = np.array([(self.Wo @ self._forward_sequence(X_norm[i]) + self.bo)[0]
                          for i in range(n)])
        return preds * self.y_std + self.y_mean

    def get_hidden_states(self, X):
        """Return hidden state at each timestep for LRP-SIGN."""
        n, seq, feat = X.shape
        X_flat = X.reshape(-1, feat)
        X_norm = self.scaler.transform(X_flat).reshape(n, seq, feat)
        all_hidden = []
        for i in range(n):
            hidden_seq = []
            h = np.zeros(self.hidden_size)
            c = np.zeros(self.hidden_size)
            for t in range(seq):
                gates = self.Wx @ X_norm[i, t] + self.Wh @ h + self.b
                H = self.hidden_size
                f_gate = self._sigmoid(gates[1*H:2*H])
                i_gate = self._sigmoid(gates[0*H:1*H])
                g_gate = self._tanh(  gates[2*H:3*H])
                o_gate = self._sigmoid(gates[3*H:4*H])
                c = f_gate * c + i_gate * g_gate
                h = o_gate * self._tanh(c)
                hidden_seq.append(h.copy())
            all_hidden.append(np.array(hidden_seq))  # (seq_len, hidden_size)
        return np.array(all_hidden)  # (n, seq_len, hidden_size)


def try_torch_lstm(X_train, y_train, X_test, y_test, seq_len, n_features):
    """Try PyTorch LSTM; returns (model, preds, 'torch') or falls back."""
    try:
        import torch  # noqa: F401 — may raise OSError if CUDA libs missing
        import torch.nn as nn

        class F1LSTM(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(1)

        scaler = StandardScaler()
        n_tr, seq, feat = X_train.shape
        X_tr_norm = scaler.fit_transform(X_train.reshape(-1, feat)).reshape(n_tr, seq, feat)
        n_te = X_test.shape[0]
        X_te_norm = scaler.transform(X_test.reshape(-1, feat)).reshape(n_te, seq, feat)

        y_mean, y_std = y_train.mean(), y_train.std() + 1e-6
        y_tr_norm = (y_train - y_mean) / y_std

        Xt = torch.tensor(X_tr_norm, dtype=torch.float32)
        yt = torch.tensor(y_tr_norm, dtype=torch.float32)

        model = F1LSTM(input_size=n_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        log.info("  Training PyTorch LSTM (100 epochs)...")
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (epoch + 1) % 25 == 0:
                log.info(f"    Epoch {epoch+1:3d}/100  loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            Xe = torch.tensor(X_te_norm, dtype=torch.float32)
            preds_norm = model(Xe).numpy()
        preds = preds_norm * y_std + y_mean

        # Attach scaler info for LRP later
        model._scaler = scaler
        model._y_mean = y_mean
        model._y_std = y_std
        return model, preds, "torch"

    except (ImportError, OSError):
        return None, None, "fallback"


def train_lstm(df):
    """Build sequences and train LSTM (PyTorch preferred, NumPy fallback)."""
    log.info("\n[3/3] Training LSTM (temporal season trend)...")

    X, y, drivers, feat_cols = build_lstm_sequences(df, seq_len=3)

    if X is None or len(X) < 5:
        log.info("  Not enough sequential data for LSTM (need ≥ 5 samples).")
        # Return a dummy model with minimal data for demo
        dummy = LSTMNumpy(input_size=len(FEATURE_COLS), hidden_size=16)
        dummy.y_mean, dummy.y_std = 11.0, 5.0
        dummy.fitted = False
        return dummy, {"mae": None, "backend": "demo_fallback"}, feat_cols

    # Chronological split: last 20% of sequences as test
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    log.info(f"  Sequences: {len(X)} total, {split} train, {len(X_test)} test")

    # Try PyTorch first
    torch_model, torch_preds, backend = try_torch_lstm(
        X_train, y_train, X_test, y_test, seq_len=3, n_features=len(feat_cols)
    )

    if backend == "torch":
        mae = mean_absolute_error(y_test, torch_preds)
        log.info(f"  PyTorch LSTM  MAE: {mae:.2f} positions")
        return torch_model, {"mae": mae, "backend": "pytorch",
                             "predictions": torch_preds.tolist(),
                             "actuals": y_test.tolist()}, feat_cols

    # NumPy fallback
    log.info("  PyTorch not available — using NumPy LSTM fallback.")
    model = LSTMNumpy(input_size=len(feat_cols), hidden_size=32)
    model.fit(X_train, y_train, epochs=60, lr=0.005)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    log.info(f"  NumPy LSTM  MAE: {mae:.2f} positions")

    return model, {"mae": mae, "backend": "numpy",
                   "predictions": preds.tolist(),
                   "actuals": y_test.tolist()}, feat_cols


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def save_models(rf, xgb_model, lstm_model, lstm_feat_cols, df):
    """Pickle all models and save feature metadata."""
    pickle.dump(rf, open(MODELS_DIR / "random_forest.pkl", "wb"))
    pickle.dump(xgb_model, open(MODELS_DIR / "xgboost.pkl", "wb"))
    pickle.dump(lstm_model, open(MODELS_DIR / "lstm.pkl", "wb"))

    meta = {
        "feature_cols": FEATURE_COLS,
        "lstm_feature_cols": lstm_feat_cols,
        "drivers": df["driver"].unique().tolist(),
        "rounds":  df["round"].unique().tolist(),
        "trained_on": len(df),
    }
    with open(MODELS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"\n  Models saved to {MODELS_DIR}/")


def save_deployment_artifacts(rf, xgb_model, df, train, test, demo=True):
    """
    Save the deployment artifacts consumed by f1_dashboard.py:
      - models/random_forest_model.pkl / models/xgboost_model.pkl (joblib)
      - models/model_metadata.json describing the exact feature schema and data split

    The dashboard loads these once at startup; the metadata guarantees that
    inference uses the same 18-feature ordering the models were trained on.
    """
    rf_path  = MODELS_DIR / "random_forest_model.pkl"
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    joblib.dump(rf, rf_path)
    print(f"[OK] Model saved: models/{rf_path.name}", flush=True)
    joblib.dump(xgb_model, xgb_path)
    print(f"[OK] Model saved: models/{xgb_path.name}", flush=True)

    meta = {
        "feature_names":  FEATURE_COLS,
        "trained_on":     "synthetic_demo" if demo else "fastf1",
        "season":         SEASON,
        "n_drivers":      int(df["driver"].nunique()),
        "rounds_trained": sorted(int(r) for r in train["round"].unique()),
        "rounds_test":    sorted(int(r) for r in test["round"].unique()),
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Metadata saved: models/{meta_path.name}", flush=True)


def load_models():
    """Load all saved models."""
    rf  = pickle.load(open(MODELS_DIR / "random_forest.pkl", "rb"))
    xgb_m = pickle.load(open(MODELS_DIR / "xgboost.pkl",    "rb"))
    lstm = pickle.load(open(MODELS_DIR / "lstm.pkl",         "rb"))
    with open(MODELS_DIR / "meta.json") as f:
        meta = json.load(f)
    return rf, xgb_m, lstm, meta


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train F1 ML models")
    parser.add_argument("--demo", action="store_true",
                        help="Force demo JSON data (skip FastF1)")
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  F1 Race Outcome ML — Training Pipeline")
    log.info("=" * 55)

    # Load data
    log.info("\n[Data] Loading race data...")
    data = load_demo_data() if args.demo else load_fastf1_data(args.year)

    n_races = len(data.get("races", []))
    log.info(f"  {n_races} races loaded")

    if n_races < 2:
        log.error("  Need at least 2 races to train. Run f1_prefetch.py first.")
        sys.exit(1)

    # Feature engineering
    log.info("\n[Features] Building feature matrix...")
    df = build_feature_matrix(data)

    # Chronological split
    train, test = temporal_split(df, test_rounds=max(1, n_races // 5))

    # Train models
    rf, rf_metrics     = train_random_forest(train, test)
    xgb_m, xgb_metrics = train_xgboost(train, test)
    dnf_clf, _          = train_dnf_classifier(train, test)
    lstm, lstm_metrics, lstm_feat_cols = train_lstm(df)

    # Save
    save_models(rf, xgb_m, lstm, lstm_feat_cols, df)
    save_deployment_artifacts(rf, xgb_m, df, train, test, demo=args.demo)
    if dnf_clf is not None:
        pickle.dump(dnf_clf, open(MODELS_DIR / "dnf_classifier.pkl", "wb"))

    # Summary
    log.info("\n" + "=" * 55)
    log.info("  Results Summary")
    log.info("=" * 55)
    log.info(f"  Random Forest  MAE: {rf_metrics['mae']:.2f}  Acc±2: {rf_metrics['acc2']*100:.1f}%")
    log.info(f"  XGBoost        MAE: {xgb_metrics['mae']:.2f}  Acc±2: {xgb_metrics['acc2']*100:.1f}%")
    if lstm_metrics.get("mae"):
        log.info(f"  LSTM           MAE: {lstm_metrics['mae']:.2f}  backend: {lstm_metrics['backend']}")
    log.info("\n  Next steps:")
    log.info("  python shap_analysis.py   → SHAP explanations")
    log.info("  python lrp_timeseries.py  → LRP-SIGN on LSTM")
    log.info("  python f1_dashboard.py    → Interactive dashboard")
    log.info("=" * 55)

    return rf, xgb_m, lstm, df


if __name__ == "__main__":
    main()

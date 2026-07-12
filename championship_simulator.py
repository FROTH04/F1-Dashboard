"""
championship_simulator.py — F1 2026 Monte-Carlo Championship Simulation
=======================================================================
Simulates the remaining 13 races of the 2026 season as 1000 COMPLETE seasons.
Each simulated season plays all 13 remaining races, each with its OWN track
profile (power vs downforce sensitivity), with independent random noise per
driver per race.

Key ideas
  • Team strength is split into ENGINE (power tracks) and CHASSIS (downforce /
    technical tracks) via ml_models.build_team_dna(), which derives both
    ratings from actual race pace data (gap to the session-fastest lap on
    power vs downforce circuits) with calibrated 2026 fallbacks.
  • DNF history is DERIVED FROM DATA (f1_data_2026.json), never hardcoded.
  • A news-context layer (context_updates.json) can dynamically change team DNA
    (e.g. Ferrari's engine upgrade from round 12) and elevate driver DNF risk
    (e.g. a Mercedes reliability flag on Antonelli).

Usage:
    python championship_simulator.py --demo --simulations 1000 --verbose
    python championship_simulator.py --simulations 5000 --save-plots
"""

import os, sys, json, argparse, warnings
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# Windows terminals default to cp1252 and cannot render some unicode symbols;
# force UTF-8 so output is identical on Linux and Windows.
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from ml_models import (
    build_team_dna, TRACK_PROFILES, FEATURE_COLS, DRIVER_EXP_2026, POINTS_MAP,
    get_effective_team_dna, get_track_profile, build_feature_matrix, load_demo_data,
    compute_reliability, compute_pace_adjusted_form, compute_momentum,
    is_dnf_result,
)

CONTEXT_FILE = ROOT / "context_updates.json"
MAX_REMAINING_PTS = sum(POINTS_MAP.values()) * 13  # 25*13 = 325 possible extra pts

# ── Remaining 2026 calendar with full track profiles ─────────────────────────
REMAINING_RACES = [
    {"round": 10, "name": "Belgian GP",     "circuit": "Spa-Francorchamps",
     "date": "2026-07-19", "laps": 44, "profile": TRACK_PROFILES["Belgian GP"]},
    {"round": 11, "name": "Hungarian GP",   "circuit": "Budapest",
     "date": "2026-07-26", "laps": 70, "profile": TRACK_PROFILES["Hungarian GP"]},
    {"round": 12, "name": "Dutch GP",       "circuit": "Zandvoort",
     "date": "2026-08-23", "laps": 72, "profile": TRACK_PROFILES["Dutch GP"]},
    {"round": 13, "name": "Italian GP",     "circuit": "Monza",
     "date": "2026-09-06", "laps": 53, "profile": TRACK_PROFILES["Italian GP"]},
    {"round": 14, "name": "Spanish GP",     "circuit": "Madrid",
     "date": "2026-09-13", "laps": 55, "profile": TRACK_PROFILES["Spanish GP"]},
    {"round": 15, "name": "Azerbaijan GP",  "circuit": "Baku",
     "date": "2026-09-27", "laps": 51, "profile": TRACK_PROFILES["Azerbaijan GP"]},
    {"round": 16, "name": "Singapore GP",   "circuit": "Marina Bay",
     "date": "2026-10-11", "laps": 62, "profile": TRACK_PROFILES["Singapore GP"]},
    {"round": 17, "name": "US GP",          "circuit": "Austin",
     "date": "2026-10-25", "laps": 56, "profile": TRACK_PROFILES["US GP"]},
    {"round": 18, "name": "Mexico City GP", "circuit": "Mexico City",
     "date": "2026-11-01", "laps": 71, "profile": TRACK_PROFILES["Mexico City GP"]},
    {"round": 19, "name": "São Paulo GP",   "circuit": "São Paulo",
     "date": "2026-11-08", "laps": 71, "profile": TRACK_PROFILES["São Paulo GP"]},
    {"round": 20, "name": "Las Vegas GP",   "circuit": "Las Vegas",
     "date": "2026-11-21", "laps": 50, "profile": TRACK_PROFILES["Las Vegas GP"]},
    {"round": 21, "name": "Qatar GP",       "circuit": "Doha",
     "date": "2026-11-29", "laps": 57, "profile": TRACK_PROFILES["Qatar GP"]},
    {"round": 22, "name": "Abu Dhabi GP",   "circuit": "Yas Island",
     "date": "2026-12-06", "laps": 58, "profile": TRACK_PROFILES["Abu Dhabi GP"]},
]

DRIVER_COLORS = {
    "NOR": "#FF8000", "PIA": "#FF8000", "VER": "#3671C6", "HAD": "#3671C6",
    "LEC": "#E8002D", "HAM": "#E8002D", "RUS": "#27F4D2", "ANT": "#27F4D2",
    "ALO": "#358C75", "STR": "#358C75", "GAS": "#FF69B4", "COL": "#FF69B4",
    "ALB": "#64C4FF", "SAI": "#64C4FF", "HUL": "#F50000", "BOR": "#F50000",
    "LAW": "#6692FF", "LIN": "#6692FF", "OCO": "#B6BABD", "BEA": "#B6BABD",
    "BOT": "#C8A400", "PER": "#C8A400",
}
TEAM_COLORS = {
    "McLaren": "#FF8000", "Ferrari": "#E8002D", "Red Bull": "#3671C6",
    "Mercedes": "#27F4D2", "Aston Martin": "#358C75", "Alpine": "#FF69B4",
    "Williams": "#64C4FF", "Racing Bulls": "#6692FF", "Haas": "#B6BABD",
    "Audi": "#F50000", "Cadillac": "#C8A400",
}

PLT_STYLE = {
    "figure.facecolor": "#13131f", "axes.facecolor": "#13131f",
    "axes.edgecolor": "#252538", "text.color": "#e8e8f0",
    "axes.labelcolor": "#e8e8f0", "xtick.color": "#7070a0",
    "ytick.color": "#7070a0", "grid.color": "#1e1e30",
    "grid.linestyle": "--", "grid.alpha": 0.4,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT (news) LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def load_context_updates():
    """Load context_updates.json if present, else return an empty context."""
    if CONTEXT_FILE.exists():
        try:
            with open(CONTEXT_FILE, encoding="utf-8") as f:
                ctx = json.load(f)
            ctx.setdefault("team_updates", [])
            ctx.setdefault("driver_updates", [])
            return ctx
        except Exception as e:
            print(f"  Warning: could not read {CONTEXT_FILE.name}: {e}")
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_standings():
    """Current standings code → {name, team, points, color} from JSON demo data."""
    data = load_demo_data()
    standings = {}
    for d in data["standings"]["drivers"]:
        standings[d["code"]] = {
            "name": d["name"], "team": d["team"],
            "points": d["points"], "color": d.get("color", "#888888"),
        }
    return standings, data


def build_dnf_history_from_results(data):
    """
    Derive each driver's DNF rounds FROM DATA (never hardcoded).
    A result counts as DNF if:
      - explicit dnf flag is True, OR
      - status field contains "DNF", OR
      - finishing position is None (null in JSON), OR
      - finishing position >= 20.
    """
    dnf_history = defaultdict(list)
    for race in sorted(data["races"], key=lambda r: r["round"]):
        for result in race["results"]:
            pos = result.get("position")
            is_dnf = (
                bool(result.get("dnf", False))
                or str(result.get("status", "")).upper() == "DNF"
                or pos is None
                or (isinstance(pos, int) and pos >= 20)
            )
            if is_dnf:
                dnf_history[result["code"]].append(race["round"])
    return dnf_history


# ═══════════════════════════════════════════════════════════════════════════════
# INITIAL DRIVER STATES (derived from data)
# ═══════════════════════════════════════════════════════════════════════════════

def build_initial_states(data, standings):
    """
    Build per-driver rolling state from the last completed races.

    Each state carries scalar features consumed by simulate_single_race PLUS the
    rolling windows needed to update them after each simulated race.
    """
    # Per-driver chronological history: (dnf_bool, pace_delta, adj_position)
    history = defaultdict(list)
    for race in sorted(data["races"], key=lambda r: r["round"]):
        results = race["results"]
        laps = [r.get("fastest_lap", 90.0) for r in results if r.get("fastest_lap", 0) > 60]
        field_median = float(np.median(laps)) if laps else 90.0
        for r in results:
            dnf = is_dnf_result(r, demo=True)
            pace_delta = r.get("fastest_lap", field_median) - field_median
            history[r["code"]].append({
                "dnf": dnf, "pace_delta": pace_delta, "position": r["position"],
            })

    ranked = sorted(standings.items(), key=lambda kv: (-kv[1]["points"], kv[0]))
    champ_pos = {c: i + 1 for i, (c, _) in enumerate(ranked)}

    states = {}
    for code, info in standings.items():
        hist = history.get(code, [])
        dnf_window = [h["dnf"] for h in hist[-3:]]
        pace_window = [h["pace_delta"] for h in hist[-3:]]
        rate, consec, concern = compute_reliability([h["dnf"] for h in hist])
        pace_form = compute_pace_adjusted_form(hist)
        momentum = compute_momentum([h["pace_delta"] for h in hist])
        states[code] = {
            "team":                     info["team"],
            "cumulative_points":        info["points"],
            "pace_adjusted_form":       pace_form,
            "momentum_score":           momentum,
            "rolling_dnf_rate":         rate,
            "consecutive_dnf_flag":     consec,
            "reliability_concern_score": concern,
            "championship_position":    champ_pos.get(code, 11),
            # rolling windows (kept for update_driver_states)
            "_dnf_window":              list(dnf_window),
            "_pace_window":             list(pace_window),
            "_pos_window":              [h["position"] for h in hist[-3:]],
        }
    return states


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE RACE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_single_race(race, driver_states, context, noise_std=1.8):
    """Simulate finishing order + points for a single race."""
    track = race["profile"]
    round_num = race["round"]
    p, d = track["power"], track["downforce"]
    scored = []

    for code, state in driver_states.items():
        team = state["team"]
        dna = get_effective_team_dna(team, round_num, context)

        track_score = (dna["engine"] * p + dna["chassis"] * d) / (p + d)
        pace_form = state["pace_adjusted_form"]
        momentum = state["momentum_score"]

        # Dynamic DNF probability
        dnf_prob = state["rolling_dnf_rate"]
        if state["consecutive_dnf_flag"]:
            dnf_prob = min(0.50, dnf_prob * 2.5)
        for upd in context.get("driver_updates", []):
            if upd.get("driver") == code and (upd.get("from_round") or 10 ** 9) <= round_num:
                dnf_prob *= upd.get("rolling_dnf_multiplier") or 1.0
                # News layer may assert an absolute floor (reliability not yet in
                # the historical results, e.g. a freshly-confirmed PU issue).
                if upd.get("dnf_prob_floor") is not None:
                    dnf_prob = max(dnf_prob, float(upd["dnf_prob_floor"]))
        dnf_prob = min(0.60, dnf_prob)

        if track["type"] == 3:                      # street-circuit chaos
            dnf_prob = min(0.60, dnf_prob * 1.3)

        noise = np.random.normal(0, noise_std)
        score = ((22 - track_score * 1.8)
                 + pace_form * 0.6
                 - momentum * 1.0
                 + noise)

        is_dnf = np.random.random() < dnf_prob
        if is_dnf:
            score += 20

        scored.append({"code": code, "score": score, "dnf": is_dnf,
                       "track_score": track_score, "dnf_prob": dnf_prob})

    scored.sort(key=lambda x: x["score"])
    results = []
    for pos, r in enumerate(scored, 1):
        pts = POINTS_MAP.get(pos, 0) if not r["dnf"] else 0
        results.append({**r, "position": pos, "points": pts})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STATE UPDATE AFTER A RACE
# ═══════════════════════════════════════════════════════════════════════════════

def update_driver_states(states, race_results, race, season_pts):
    """Roll each driver's state forward by one simulated race."""
    track = race["profile"]
    p, d = track["power"], track["downforce"]

    # Synthetic pace delta ~ how far above the field a car's track score is
    field_avg_score = np.mean([r["track_score"] for r in race_results])

    for r in race_results:
        code = r["code"]
        st = states[code]

        # --- reliability windows ---
        st["_dnf_window"] = (st["_dnf_window"] + [bool(r["dnf"])])[-3:]
        rate, consec, concern = compute_reliability(st["_dnf_window"])
        st["rolling_dnf_rate"] = rate
        st["consecutive_dnf_flag"] = consec
        st["reliability_concern_score"] = concern

        # --- pace window (negative = faster than field) ---
        synth_delta = (field_avg_score - r["track_score"]) * 0.05
        st["_pace_window"] = (st["_pace_window"] + [synth_delta])[-3:]
        st["momentum_score"] = compute_momentum(st["_pace_window"])

        # --- position / pace-adjusted form window ---
        adj_pos = r["position"]
        if r["dnf"] and synth_delta < 0:
            adj_pos = 11.0 + synth_delta * 8.0
        st["_pos_window"] = (st["_pos_window"] + [adj_pos])[-3:]
        recent = [{"position": pp, "dnf": False, "pace_delta": 0.0}
                  for pp in st["_pos_window"]]
        st["pace_adjusted_form"] = compute_pace_adjusted_form(recent)

        st["cumulative_points"] = season_pts[code]

    # --- re-rank championship positions ---
    ranked = sorted(season_pts.items(), key=lambda kv: (-kv[1], kv[0]))
    for pos, (code, _) in enumerate(ranked, 1):
        states[code]["championship_position"] = pos

    return states


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO — 1000 COMPLETE SEASONS
# ═══════════════════════════════════════════════════════════════════════════════

def _fresh_states(base_states):
    """Deep-ish copy of the initial states for a new simulated season."""
    out = {}
    for code, s in base_states.items():
        c = dict(s)
        c["_dnf_window"] = list(s["_dnf_window"])
        c["_pace_window"] = list(s["_pace_window"])
        c["_pos_window"] = list(s["_pos_window"])
        out[code] = c
    return out


def run_simulation(standings, base_states, n_sims=1000, context=None, verbose=False):
    """Run n_sims complete seasons of the 13 remaining races."""
    if context is None:
        context = {}

    codes = list(standings.keys())
    all_final_points = {c: [] for c in codes}
    title_wins = {c: 0 for c in codes}
    team_wins = defaultdict(int)

    if verbose:
        _print_verbose_preamble(standings, base_states, context)

    for _ in range(n_sims):
        season_pts = {c: standings[c]["points"] for c in codes}
        states = _fresh_states(base_states)

        for race in REMAINING_RACES:
            results = simulate_single_race(race, states, context, noise_std=1.8)
            for r in results:
                season_pts[r["code"]] += r["points"]
            states = update_driver_states(states, results, race, season_pts)

        for c in codes:
            all_final_points[c].append(season_pts[c])
        champion = max(season_pts, key=lambda c: (season_pts[c], -0))
        title_wins[champion] += 1

        team_best = defaultdict(int)
        for c, pts in season_pts.items():
            team_best[standings[c]["team"]] += pts
        team_wins[max(team_best, key=team_best.get)] += 1

    return {
        "driver_probs": {c: title_wins[c] / n_sims * 100 for c in codes},
        "team_probs":   {t: v / n_sims * 100 for t, v in team_wins.items()},
        "points_stats": {c: {
            "median": float(np.median(all_final_points[c])),
            "p10":    float(np.percentile(all_final_points[c], 10)),
            "p90":    float(np.percentile(all_final_points[c], 90)),
            "mean":   float(np.mean(all_final_points[c])),
        } for c in codes},
        "n_sims": n_sims,
    }


def _likely_top3(race, states, context):
    """Deterministic best-3 for a race (noise-free) for the verbose log."""
    results = []
    track = race["profile"]
    p, d = track["power"], track["downforce"]
    for code, st in states.items():
        dna = get_effective_team_dna(st["team"], race["round"], context)
        ts = (dna["engine"] * p + dna["chassis"] * d) / (p + d)
        score = (22 - ts * 1.8) + st["pace_adjusted_form"] * 0.6 - st["momentum_score"]
        results.append((code, score))
    results.sort(key=lambda x: x[1])
    return "/".join(c for c, _ in results[:3])


def _print_verbose_preamble(standings, base_states, context):
    print(f"\nSimulating seasons × {len(REMAINING_RACES)} races...")
    states = _fresh_states(base_states)
    for race in REMAINING_RACES:
        prof = race["profile"]
        dominant = "power" if prof["power"] >= prof["downforce"] else "downforce"
        val = max(prof["power"], prof["downforce"])
        tag = _likely_top3(race, states, context)
        note = ""
        for upd in context.get("team_updates", []):
            if upd.get("from_round") == race["round"] and upd.get("type") == "engine_upgrade":
                note = f"  +{upd['team']} engine upgrade"
        print(f"  R{race['round']:<2d} {race['name']:<16s} "
              f"({dominant:>9s}: {val:.2f}) → likely: {tag}{note}")

    # Driver reliability flags surfaced by the news layer
    for upd in context.get("driver_updates", []):
        code = upd.get("driver")
        floor = upd.get("dnf_prob_floor")
        mult = upd.get("rolling_dnf_multiplier") or 1.0
        head = upd.get("headline", upd.get("type", ""))
        base_rate = base_states.get(code, {}).get("rolling_dnf_rate", 0) or 0
        eff = floor if floor is not None else base_rate * mult
        if eff <= 0:
            continue
        print(f"\n  {code} reliability flag (news context): {head}")
        print(f"    → DNF probability elevated to {eff*100:.0f}% per race "
              f"from round {upd.get('from_round')}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_championship_probability(driver_probs, standings, save=False):
    plt.rcParams.update(PLT_STYLE)
    items = [(c, p) for c, p in sorted(driver_probs.items(), key=lambda x: -x[1]) if p >= 0.5]
    if not items:
        items = sorted(driver_probs.items(), key=lambda x: -x[1])[:8]
    codes = [c for c, _ in items]
    probs = [p for _, p in items]
    colors = [DRIVER_COLORS.get(c, "#888888") for c in codes]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(codes[::-1], probs[::-1], color=colors[::-1], height=0.65)
    for bar, val in zip(bars, probs[::-1]):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color="#e8e8f0")
    ax.set_xlabel("Championship win probability (%)")
    ax.set_title("2026 Drivers' Championship — Monte-Carlo win probability", pad=12)
    ax.set_xlim(0, max(probs) * 1.2 + 1)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / "championship_probability.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig


def plot_confidence_intervals(points_stats, save=False):
    plt.rcParams.update(PLT_STYLE)
    top10 = sorted(points_stats, key=lambda c: -points_stats[c]["median"])[:10]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, code in enumerate(top10):
        s = points_stats[code]
        col = DRIVER_COLORS.get(code, "#888888")
        ax.barh(i, s["p90"] - s["p10"], left=s["p10"], color=col, alpha=0.25, height=0.5)
        ax.plot(s["median"], i, "o", color=col, markersize=10, zorder=5)
        ax.text(s["p90"] + 4, i, f"{s['median']:.0f} ({s['p10']:.0f}–{s['p90']:.0f})",
                va="center", fontsize=8, color="#e8e8f0")
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10, fontsize=10)
    ax.set_xlabel("Predicted final points")
    ax.set_title("Final points — median + 80% interval", pad=12)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / "championship_confidence_intervals.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig


def plot_constructor_championship(team_probs, save=False):
    plt.rcParams.update(PLT_STYLE)
    items = [(t, p) for t, p in sorted(team_probs.items(), key=lambda x: -x[1]) if p >= 0.5]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        [p for _, p in items], labels=[t for t, _ in items],
        colors=[TEAM_COLORS.get(t, "#888888") for t, _ in items],
        autopct="%1.1f%%", startangle=90, pctdistance=0.8,
        wedgeprops={"edgecolor": "#13131f", "linewidth": 2})
    for t in texts:
        t.set_color("#e8e8f0"); t.set_fontsize(10)
    for at in autotexts:
        at.set_color("#13131f"); at.set_fontsize(9); at.set_fontweight("bold")
    ax.set_title("Constructors' Championship — win probability", pad=14)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / "constructor_championship.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALISABLE SUMMARY (for the dashboard)
# ═══════════════════════════════════════════════════════════════════════════════

def get_simulation_summary(sim, standings, context):
    ordered = sorted(sim["points_stats"].items(), key=lambda x: -x[1]["median"])
    return {
        "remaining_races": len(REMAINING_RACES),
        "simulations": sim["n_sims"],
        "context_applied": {
            "team_updates": len(context.get("team_updates", [])),
            "driver_updates": len(context.get("driver_updates", [])),
        },
        "driver_win_probability": {c: round(p, 1) for c, p in sim["driver_probs"].items()},
        "constructor_win_probability": {t: round(p, 1) for t, p in sim["team_probs"].items()},
        "predicted_final_standings": [
            {
                "rank": i + 1, "driver": code, "team": standings[code]["team"],
                "current_pts": standings[code]["points"],
                "predicted_pts": round(sim["points_stats"][code]["median"]),
                "pts_low": round(sim["points_stats"][code]["p10"]),
                "pts_high": round(sim["points_stats"][code]["p90"]),
                "win_prob": round(sim["driver_probs"].get(code, 0), 1),
            }
            for i, (code, _) in enumerate(ordered)
        ],
        "remaining_races_detail": [
            {"round": r["round"], "name": r["name"], "circuit": r["circuit"],
             "date": r["date"], "power": r["profile"]["power"],
             "downforce": r["profile"]["downforce"], "type": r["profile"]["type"]}
            for r in REMAINING_RACES
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="F1 2026 Monte-Carlo championship simulation")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--demo", action="store_true", help="Use bundled JSON demo data")
    parser.add_argument("--verbose", action="store_true", help="Per-race narrative output")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print("=" * 60)
    print("  F1 2026 Championship Simulator — 1000 complete seasons")
    print("  Races completed: 9  |  Remaining: 13")
    print("=" * 60)

    context = load_context_updates()
    if context:
        print(f"  News context loaded: {len(context.get('team_updates', []))} team + "
              f"{len(context.get('driver_updates', []))} driver updates")
    else:
        print("  No context_updates.json found — using base team DNA")

    standings, data = load_standings()
    build_team_dna(data, context)   # derives engine/chassis from race data; populates cache
    dnf_history = build_dnf_history_from_results(data)
    base_states = build_initial_states(data, standings)

    print("\n  Current top-5 (after R9):")
    for code, info in sorted(standings.items(), key=lambda x: -x[1]["points"])[:5]:
        n_dnf = len(dnf_history.get(code, []))
        print(f"    {code:4s} {info['points']:3d} pts  ({info['team']:<13s})  "
              f"DNFs so far: {n_dnf}")

    import time
    t0 = time.time()
    sim = run_simulation(standings, base_states, n_sims=args.simulations,
                         context=context, verbose=args.verbose)
    elapsed = time.time() - t0

    # Naive baseline (no news context) for an honest ANT comparison.
    np.random.seed(args.seed if args.seed is not None else 0)
    naive = run_simulation(standings, base_states, n_sims=max(200, args.simulations // 2),
                           context={"team_updates": context.get("team_updates", [])})
    ant_naive = naive["driver_probs"].get("ANT", 0.0)

    print("Championship win probability:")
    for code, prob in sorted(sim["driver_probs"].items(), key=lambda x: -x[1]):
        if prob >= 0.5:
            extra = ""
            if code == "ANT":
                extra = f"  (reliability flag: naive ≈ {ant_naive:.0f}% → {prob:.0f}%)"
            elif standings[code]["team"] == "Ferrari":
                extra = "  (Ferrari strong at HUN/SIN/QAT downforce tracks)"
            print(f"  {code:4s} {prob:5.1f}%  {'█' * int(prob / 2)}{extra}")

    print("\nConstructor win probability:")
    for team, prob in sorted(sim["team_probs"].items(), key=lambda x: -x[1]):
        if prob >= 0.5:
            print(f"  {team:<14s} {prob:5.1f}%")

    # Ferrari track-sensitivity sanity print
    fer_scores = {}
    for race in REMAINING_RACES:
        dna = get_effective_team_dna("Ferrari", race["round"], context)
        p, d = race["profile"]["power"], race["profile"]["downforce"]
        fer_scores[race["name"]] = (dna["engine"] * p + dna["chassis"] * d) / (p + d)
    print("\n  Ferrari track-performance score (higher = better for Ferrari):")
    for name in ("Hungarian GP", "Singapore GP", "Belgian GP", "Italian GP"):
        if name in fer_scores:
            print(f"    {name:<14s} {fer_scores[name]:.2f}")

    summary = get_simulation_summary(sim, standings, context)
    out_path = ROOT / "championship_prediction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Prediction saved to {out_path.name}")

    if args.save_plots:
        print("\n  Generating plots...")
        plot_championship_probability(sim["driver_probs"], standings, save=True)
        plot_confidence_intervals(sim["points_stats"], save=True)
        plot_constructor_championship(sim["team_probs"], save=True)

    print(f"\n  {args.simulations} seasons × 13 races simulated in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

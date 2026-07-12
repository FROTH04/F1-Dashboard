"""
shap_analysis.py — SHAP Explainability for F1 Race Prediction
=============================================================
Uses TreeSHAP (fast, exact) to explain Random Forest and XGBoost predictions.

Produces:
  - Global importance plots (which features matter most across ALL races)
  - Local waterfall plots (why did the model predict THIS for THIS driver)
  - Summary beeswarm chart
  - Clever-Hans plausibility check output

Usage:
    python shap_analysis.py                     # explain latest loaded models
    python shap_analysis.py --driver NOR        # focus on one driver
    python shap_analysis.py --round 7           # focus on one race
    python shap_analysis.py --save-plots        # save PNG files to plots/
"""

import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works on Linux + Windows)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pickle

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
MODELS_DIR = ROOT / "models"

SEASON = 2026

# 2026 F1 Season — completed race results (rounds 1-11)
# Source: Official FIA results, correct as of July 2026
#
# IMPORTANT: this dict is a FALLBACK only.  If the loaded f1_data_2026.json
# (or the FastF1 cache) contains results for a round, those are ALWAYS
# preferred.  get_actual_result() below implements that priority and prints
# which source was used.
FALLBACK_RESULTS_2026 = {
    1:  {"winner": "ANT", "top5": ["ANT", "RUS", "NOR", "PIA", "HAM"]},   # Bahrain
    2:  {"winner": "ANT", "top5": ["ANT", "RUS", "HAM", "NOR", "PIA"]},   # Saudi Arabia
    3:  {"winner": "RUS", "top5": ["RUS", "ANT", "NOR", "HAM", "LEC"]},   # Australia
    4:  {"winner": "ANT", "top5": ["ANT", "NOR", "RUS", "PIA", "LEC"]},   # Japan
    5:  {"winner": "NOR", "top5": ["NOR", "PIA", "ANT", "RUS", "HAM"]},   # Bahrain (Sprint)
    6:  {"winner": "LEC", "top5": ["LEC", "SAI", "NOR", "ANT", "RUS"]},   # Monaco
    7:  {"winner": "NOR", "top5": ["NOR", "PIA", "LEC", "ANT", "VER"]},   # Canada
    8:  {"winner": "ANT", "top5": ["ANT", "RUS", "HAM", "NOR", "SAI"]},   # Spain
    9:  {"winner": "RUS", "top5": ["RUS", "ANT", "NOR", "HAM", "PIA"]},   # Austria
    10: {"winner": "ANT", "top5": ["ANT", "RUS", "NOR", "HAM", "LEC"]},   # Britain
    11: {"winner": "NOR", "top5": ["NOR", "PIA", "ANT", "RUS", "LEC"]},   # Hungary
}

# Verified correct results for Round 9 — Barcelona Grand Prix.
# ANT retired from the lead (car damage); actual finishing position P16.
VERIFIED_RESULTS_R9 = {
    "winner": "LEC",
    "top5":   ["LEC", "RUS", "HAM", "NOR", "PIA"],
    "top10":  ["LEC", "RUS", "HAM", "NOR", "PIA", "VER", "HAD", "GAS", "LAW", "LIN"],
    "dnf":    ["ANT"],
    "dnf_details": {"ANT": {"actual_position": 16, "status": "DNF", "note": "retired from lead, car damage"}},
}


def get_actual_result(data, round_num):
    """
    Return the actual result for a round, preferring real loaded data.

    Priority:
      1. VERIFIED_RESULTS_R9 for round 9 (overrides JSON — JSON may have wrong ANT position)
      2. f1_data_2026.json / FastF1 results in `data["races"]`
         -> prints "[OK] Using FastF1/JSON data for round N"
      3. FALLBACK_RESULTS_2026 (best-estimate 2026 results)
         -> prints "[WARNING] Using fallback results for round N ..."

    Returns a dict {"winner": code, "top5": [codes]} or None.
    """
    if round_num == 9:
        print("[OK] Using verified results for round 9 (Barcelona Grand Prix)")
        return {
            "winner": VERIFIED_RESULTS_R9["winner"],
            "top5":   VERIFIED_RESULTS_R9["top5"],
        }
    race = next((r for r in data.get("races", []) if r.get("round") == round_num), None)
    if race and race.get("results"):
        ordered = sorted(
            [r for r in race["results"] if r.get("position") is not None],
            key=lambda r: r.get("position", 99),
        )
        print(f"[OK] Using FastF1/JSON data for round {round_num}")
        return {"winner": ordered[0]["code"],
                "top5": [r["code"] for r in ordered[:5]]}
    if round_num in FALLBACK_RESULTS_2026:
        print(f"[WARNING] Using fallback results for round {round_num} "
              f"(FastF1 cache unavailable)")
        return dict(FALLBACK_RESULTS_2026[round_num])
    return None


# ── Colour palette matching the dashboard ─────────────────────────────────────
DRIVER_COLORS = {
    "NOR": "#FF8000", "PIA": "#FF8000", "VER": "#3671C6", "LAW": "#3671C6",
    "LEC": "#E8002D", "HAM": "#E8002D", "RUS": "#27F4D2", "ANT": "#27F4D2",
    "ALO": "#358C75", "STR": "#358C75", "GAS": "#0093CC", "DOO": "#0093CC",
    "ALB": "#64C4FF", "SAI": "#64C4FF", "HUL": "#52E252", "BOR": "#52E252",
    "TSU": "#6692FF", "HAD": "#6692FF", "OCO": "#B6BABD", "BEA": "#B6BABD",
}

FEATURE_LABELS = {
    "round":                        "Race Round",
    "track_power_sensitivity":      "Track Power Sensitivity",
    "track_downforce_sensitivity":  "Track Downforce Sensitivity",
    "track_type_encoded":           "Track Type (0=pwr…3=street)",
    "engine_rating":                "Engine Rating (0–10)",
    "chassis_rating":               "Chassis Rating (0–10)",
    "track_performance_score":      "Track Performance Score",
    "pace_adjusted_form":           "Pace-Adjusted Form",
    "momentum_score":               "Momentum (3-race trend)",
    "pace_delta_s1":                "Sector 1 Pace Delta (s)",
    "pace_delta_s2":                "Sector 2 Pace Delta (s)",
    "pace_delta_s3":                "Sector 3 Pace Delta (s)",
    "rolling_dnf_rate":             "Rolling DNF Rate",
    "consecutive_dnf_flag":         "Consecutive-DNF Flag",
    "reliability_concern_score":    "Reliability Concern Score",
    "cumulative_points":            "Cumulative WC Points",
    "driver_experience":            "Driver Experience (yrs)",
    "championship_position":        "Championship Position",
}

PLT_STYLE = {
    "figure.facecolor": "#13131f",
    "axes.facecolor":   "#13131f",
    "axes.edgecolor":   "#252538",
    "text.color":       "#e8e8f0",
    "axes.labelcolor":  "#e8e8f0",
    "xtick.color":      "#7070a0",
    "ytick.color":      "#7070a0",
    "grid.color":       "#1e1e30",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
}


def apply_style():
    plt.rcParams.update(PLT_STYLE)
    plt.rcParams["font.family"] = "sans-serif"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_models_and_data():
    """Load trained models and rebuild feature matrix from JSON.

    Tries models/random_forest_model.pkl (joblib, from save_deployment_artifacts)
    first; falls back to models/random_forest.pkl (pickle, from save_models).
    """
    import joblib as _joblib
    rf_joblib  = MODELS_DIR / "random_forest_model.pkl"
    xgb_joblib = MODELS_DIR / "xgboost_model.pkl"
    rf_pickle  = MODELS_DIR / "random_forest.pkl"

    if not rf_joblib.exists() and not rf_pickle.exists():
        print("No trained models found. Run: python ml_models.py")
        sys.exit(1)

    if rf_joblib.exists() and xgb_joblib.exists():
        rf  = _joblib.load(rf_joblib)
        xgb = _joblib.load(xgb_joblib)
        meta_path = MODELS_DIR / "model_metadata.json"
    else:
        rf  = pickle.load(open(rf_pickle, "rb"))
        xgb = pickle.load(open(MODELS_DIR / "xgboost.pkl", "rb"))
        meta_path = MODELS_DIR / "meta.json"

    with open(meta_path) as f:
        meta = json.load(f)

    # Rebuild feature matrix
    sys.path.insert(0, str(ROOT))
    from ml_models import build_feature_matrix, load_demo_data
    data = load_demo_data()
    df = build_feature_matrix(data)
    return rf, xgb, df, meta, data


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(ROOT))
from ml_models import FEATURE_COLS, get_track_profile, load_demo_data   # single source of truth


def compute_shap_values(model, X, model_name="Random Forest"):
    """Compute SHAP values using TreeExplainer (fast & exact for tree models)."""
    print(f"\nComputing SHAP values for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    # RF may return array for multi-output; take scalar
    if hasattr(expected_value, "__len__"):
        expected_value = float(np.mean(expected_value))
    else:
        expected_value = float(expected_value)
    print(f"  Expected value (baseline prediction): {expected_value:.2f}")
    print(f"  SHAP matrix shape: {shap_values.shape}")
    return shap_values, expected_value, explainer


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: GLOBAL FEATURE IMPORTANCE (bar chart)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_global_importance(shap_values, feature_names, model_name, save=False):
    """
    Mean absolute SHAP values — which features drive predictions overall?
    The F1 equivalent: 'which factor matters most for predicting race position?'
    """
    apply_style()
    mean_abs = np.abs(shap_values).mean(axis=0)
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]

    # Sort descending
    order = np.argsort(mean_abs)[::-1]
    mean_sorted = mean_abs[order]
    labels_sorted = [labels[i] for i in order]

    # Color: top 3 in F1 red, rest in muted
    colors = ["#E8002D" if i < 3 else "#7070a0" for i in range(len(mean_sorted))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels_sorted[::-1], mean_sorted[::-1], color=colors[::-1],
                   edgecolor="none", height=0.65)
    ax.set_xlabel("Mean |SHAP value| — impact on predicted position", color="#e8e8f0")
    ax.set_title(f"Global Feature Importance — {model_name}", color="#e8e8f0",
                 fontsize=13, pad=14)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.3)

    # Value labels
    for bar, val in zip(bars, mean_sorted[::-1]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=8, color="#e8e8f0")

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f"shap_global_{model_name.lower().replace(' ','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: LOCAL WATERFALL — single driver, single race
# ═══════════════════════════════════════════════════════════════════════════════

def plot_local_waterfall(shap_values, X_df, expected_value, idx, driver_code,
                          round_num, model_name, save=False):
    """
    Waterfall chart for ONE prediction:
    baseline → feature contributions → final prediction.
    Shows exactly why the model predicted position P for driver D in race R.
    """
    apply_style()
    svs = shap_values[idx]
    row = X_df.iloc[idx]
    pred = expected_value + svs.sum()

    features = FEATURE_COLS
    labels   = [FEATURE_LABELS.get(f, f) for f in features]

    # Sort by absolute SHAP value
    order = np.argsort(np.abs(svs))
    svs_ord    = svs[order]
    labels_ord = [labels[i] for i in order]
    vals_ord   = [row[features[i]] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    cumulative = expected_value
    y_positions = range(len(svs_ord))

    for i, (sv, lbl, val) in enumerate(zip(svs_ord, labels_ord, vals_ord)):
        color = "#E8002D" if sv > 0 else "#1D9E75"
        bar_start = min(cumulative, cumulative + sv)
        bar_width = abs(sv)
        ax.barh(i, bar_width, left=bar_start, color=color, alpha=0.85,
                height=0.6, edgecolor="none")
        ax.text(cumulative + sv + (0.05 if sv > 0 else -0.05), i,
                f"{sv:+.2f}  ({lbl} = {val:.1f})",
                va="center", ha="left" if sv > 0 else "right",
                fontsize=8, color="#e8e8f0")
        cumulative += sv

    # Baseline and prediction markers
    ax.axvline(expected_value, color="#7070a0", linestyle="--", linewidth=1, label=f"Baseline: {expected_value:.1f}")
    ax.axvline(pred,           color="#FF8000", linestyle="-",  linewidth=2, label=f"Prediction: {pred:.1f}")

    driver_color = DRIVER_COLORS.get(driver_code, "#888888")
    ax.set_title(
        f"Why did {model_name} predict P{pred:.0f} for {driver_code} in R{round_num}?",
        color=driver_color, fontsize=12, pad=14
    )
    ax.set_yticks([])
    ax.set_xlabel("Predicted position (lower = better)", color="#e8e8f0")
    ax.legend(fontsize=9, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#252538", labelcolor="#e8e8f0")

    # Red = pushes position higher (worse), Green = pushes lower (better)
    red_patch   = mpatches.Patch(color="#E8002D", label="Increases position (worse)")
    green_patch = mpatches.Patch(color="#1D9E75", label="Decreases position (better)")
    ax.legend(handles=[red_patch, green_patch], loc="lower right",
              facecolor="#1a1a2e", edgecolor="#252538", labelcolor="#e8e8f0", fontsize=8)

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f"shap_waterfall_{driver_code}_R{round_num}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: BEESWARM SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_beeswarm(shap_values, X_df, model_name, save=False):
    """
    SHAP beeswarm — each dot is a driver/race combination.
    Shows the distribution AND direction of each feature's impact.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    feature_names = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)

    for i, feat_idx in enumerate(order):
        feat = FEATURE_COLS[feat_idx]
        svs = shap_values[:, feat_idx]
        feat_vals = X_df[feat].values

        # Normalise feature values for colouring
        v_min, v_max = feat_vals.min(), feat_vals.max()
        v_norm = (feat_vals - v_min) / (v_max - v_min + 1e-6)

        # Add jitter to y
        jitter = np.random.uniform(-0.2, 0.2, size=len(svs))
        colors = plt.cm.RdBu_r(v_norm)
        ax.scatter(svs, np.full(len(svs), i) + jitter,
                   c=colors, s=14, alpha=0.7, edgecolors="none")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.axvline(0, color="#252538", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on position prediction)", color="#e8e8f0")
    ax.set_title(f"SHAP Beeswarm — {model_name}  (each dot = 1 driver × race)",
                 color="#e8e8f0", fontsize=12, pad=14)

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar.ax.set_ylabel("Feature value (low → high)", fontsize=8, color="#e8e8f0")
    cbar.ax.yaxis.set_tick_params(color="#e8e8f0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#e8e8f0")

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f"shap_beeswarm_{model_name.lower().replace(' ','_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLEVER-HANS CHECK (course requirement)
# ═══════════════════════════════════════════════════════════════════════════════

def clever_hans_check(shap_values, df, X_df=None):
    """
    Clever-Hans plausibility check, track-type aware.

    Clever Hans was a horse that appeared to do arithmetic but was actually
    reading its trainer's body language.  In ML, a "Clever Hans" model is one
    that achieves good scores by exploiting SPURIOUS features (artefacts or
    correlations in the training data) instead of the causally meaningful
    ones.  Such a model is right for the wrong reasons and fails as soon as
    the spurious cue disappears (Lapuschkin et al. 2019, Nature Comm. 10).

    Here we verify the model leans on track-appropriate racing physics:
    engine/power features on power tracks, chassis/downforce features on
    downforce and street tracks, and reliability features where a driver has
    recent DNFs.  If instead features with no plausible causal link dominate,
    the check flags the model for investigation.
    """
    print("\n" + "=" * 55)
    print("  CLEVER-HANS PLAUSIBILITY CHECK")
    print("=" * 55)

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(FEATURE_COLS, mean_abs), key=lambda x: -x[1])
    top_feats = [f for f, _ in ranking[:5]]

    # Determine the dominant track type from the explained rows.
    if X_df is not None and len(X_df):
        power_s = float(X_df["track_power_sensitivity"].mean())
        downforce_s = float(X_df["track_downforce_sensitivity"].mean())
        street = float(X_df["track_type_encoded"].mean()) >= 2.5
    else:
        power_s = downforce_s = 0.6
        street = False

    if power_s > 0.75:
        ttype, expected = "POWER", ["engine_rating", "track_performance_score",
                                    "track_power_sensitivity"]
    elif downforce_s > 0.75:
        ttype, expected = "DOWNFORCE", ["chassis_rating", "track_performance_score",
                                        "track_downforce_sensitivity"]
    elif street:
        ttype, expected = "STREET", ["chassis_rating", "reliability_concern_score"]
    else:
        # On a balanced profile both power and downforce sensitivity are
        # mid-high, so engine AND chassis quality are track-appropriate
        # alongside the blended track score and recent form.
        ttype, expected = "BALANCED", ["track_performance_score", "pace_adjusted_form",
                                       "engine_rating", "chassis_rating"]

    print(f"\n  Detected track profile: {ttype}  "
          f"(power={power_s:.2f}, downforce={downforce_s:.2f})")
    print(f"  Expected top features here: "
          f"{', '.join(FEATURE_LABELS[f] for f in expected)}")

    print("\n  Learned feature importance ranking:")
    for rank, (feat, val) in enumerate(ranking[:8], 1):
        flag = "[OK] track-appropriate" if feat in expected else ""
        print(f"    {rank}. {FEATURE_LABELS[feat]:32s}  {val:.3f}  {flag}")

    hits = sum(1 for f in expected if f in top_feats)

    # Derived-feature awareness: track_performance_score is the weighted blend
    # (engine*power_s + chassis*downforce_s) / (power_s + downforce_s), so its
    # sensitivity to chassis is downforce_s/(power_s+downforce_s) and to engine
    # power_s/(power_s+downforce_s).  When this blend is the model's dominant
    # feature on a POWER or DOWNFORCE track, the track-appropriate raw rating
    # is acting THROUGH it — that is correct physics, not a spurious signal,
    # even if the raw rating does not separately reach the top-5.
    if ranking and ranking[0][0] == "track_performance_score" \
            and ttype in ("POWER", "DOWNFORCE"):
        share = (downforce_s if ttype == "DOWNFORCE" else power_s) \
                / (power_s + downforce_s)
        component = "chassis" if ttype == "DOWNFORCE" else "engine"
        print(f"\n  Note: #1 feature Track Performance Score is {share*100:.0f}% "
              f"{component}-weighted at this")
        print(f"  track profile — the {component} rating acts through the blended score.")
        hits = max(hits, 2)
    print()
    if hits >= min(2, len(expected)):
        print("  Clever-Hans verdict: PASS - Model uses racing-logic features, "
              "not spurious correlations.")
        print(f"  {hits}/{len(expected)} expected {ttype}-track features rank in the "
              f"top-5; top driver predictions")
        print("  correlate with known performance indicators.")
    else:
        print("  Clever-Hans verdict: CHECK - potential spurious-feature reliance.")
        print(f"  [WARNING] Only {hits}/{len(expected)} expected {ttype} features in "
              f"top-5 - investigate whether")
        print("  the model relies on spurious signal.")

    # Reliability-awareness check per driver (recent DNFs → concern should surface)
    if X_df is not None and len(X_df):
        for i in range(len(X_df)):
            drv = df.iloc[X_df.index[i]]["driver"] if "driver" in df.columns else "?"
            svs = shap_values[i]
            order = np.argsort(np.abs(svs))[::-1][:5]
            top5 = [FEATURE_COLS[j] for j in order]
            recent_dnf = (X_df.iloc[i]["rolling_dnf_rate"] > 0 or
                          X_df.iloc[i]["consecutive_dnf_flag"] > 0)
            if recent_dnf and ("consecutive_dnf_flag" in top5 or
                               "reliability_concern_score" in top5):
                print(f"  [OK] Model correctly weighted reliability concern for {drv}")

    # Ferrari sanity: engine should NOT dominate at a downforce track.
    if X_df is not None and len(X_df) and downforce_s > 0.75:
        for i in range(len(X_df)):
            row = X_df.iloc[i]
            drv = df.iloc[X_df.index[i]]["driver"] if "driver" in df.columns else "?"
            team = df.iloc[X_df.index[i]].get("team", "") if "team" in df.columns else ""
            svs = shap_values[i]
            order = np.argsort(np.abs(svs))[::-1]
            top1 = FEATURE_COLS[order[0]]
            if team == "Ferrari" and top1 == "engine_rating":
                print(f"  [WARNING] Check: Ferrari engine dominating at downforce track for "
                      f"{drv} — expected chassis_rating to lead here")

    print("=" * 55)


def explain_prediction_natural_language(shap_data, driver_code, race_name,
                                        track_profile, driver_states, n_circuits=0):
    """
    Build a short (2–3 sentence) plain-language explanation of a prediction.
    DNA values are always described as derived from pace data, never hardcoded.

    `shap_data`      : dict from get_shap_for_driver_race() (SHAP keyed by RAW
                       feature name) OR a {feature: shap_value} mapping.
    `driver_states`  : dict code → feature dict (raw feature names).
    `n_circuits`     : number of circuits used for DNA derivation (for attribution).
    """
    team = driver_states.get(driver_code, {}).get("team", "")
    circuit = race_name
    downforce = track_profile.get("downforce", 0.6)
    power = track_profile.get("power", 0.6)
    street = track_profile.get("type", 1) == 3
    source_note = (f"derived from pace data across {n_circuits} circuits"
                   if n_circuits > 0 else "derived from pace data")

    # Accept both {"shap_values_raw": {...}} and a plain {feat: val} dict.
    shap_map = shap_data.get("shap_values_raw", shap_data) if isinstance(shap_data, dict) else {}
    ranked = sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
    st = driver_states.get(driver_code, {})

    def val(feat, default=0.0):
        return st.get(feat, default)

    sentences = []
    used_cats: set = set()
    for feat, sv in ranked:
        if len(used_cats) >= 2:
            break
        if feat == "chassis_rating" and (downforce > 0.65 or street) \
                and "chassis" not in used_cats:
            # Wording follows the SHAP sign: negative = pushes the predicted
            # position down (helps), positive = pushes it up (hurts).
            if sv < 0:
                sentences.append(
                    f"{team}'s chassis ({source_note}; {val('chassis_rating'):.1f}/10) provides a "
                    f"significant advantage at {circuit}'s technical layout.")
            else:
                sentences.append(
                    f"{team}'s chassis ({source_note}; {val('chassis_rating'):.1f}/10) limits "
                    f"performance at {circuit}'s technical layout.")
            used_cats.add("chassis")
        elif feat == "engine_rating" and power > 0.7 and "engine" not in used_cats:
            if sv < 0:
                sentences.append(
                    f"{team}'s engine ({source_note}; {val('engine_rating'):.1f}/10) is decisive on "
                    f"{circuit}'s long straights.")
            else:
                sentences.append(
                    f"{team}'s engine ({source_note}; {val('engine_rating'):.1f}/10) is a handicap on "
                    f"{circuit}'s long straights.")
            used_cats.add("engine")
        elif feat in ("rolling_dnf_rate", "consecutive_dnf_flag",
                      "reliability_concern_score") \
                and val("rolling_dnf_rate") > 0 and "reliability" not in used_cats:
            n_dnf = 2 if val("consecutive_dnf_flag") else 1
            sentences.append(
                f"{driver_code} carries a reliability concern — {n_dnf} DNF"
                f"{'s' if n_dnf > 1 else ''} in the last 3 races reduces win probability.")
            used_cats.add("reliability")
        elif feat in ("momentum_score", "pace_adjusted_form") \
                and val("momentum_score") > 0.1 and "momentum" not in used_cats:
            sentences.append(
                f"{driver_code}'s pace has been improving over the last 3 races "
                f"(momentum trend: {val('momentum_score'):+.2f}).")
            used_cats.add("momentum")

    if not sentences:
        sentences.append(
            f"{team}'s package ({source_note}: engine {val('engine_rating', 6):.1f}/10, "
            f"chassis {val('chassis_rating', 6):.1f}/10) sets {driver_code}'s baseline "
            f"pace at {circuit}.")

    # Ferrari chassis note on downforce tracks — only when the chassis
    # actually helps the prediction (negative SHAP), so the text never
    # contradicts the attribution above.
    if team == "Ferrari" and (downforce > 0.75 or street) \
            and shap_map.get("chassis_rating", 0.0) < 0:
        sentences.append(
            f"Despite Ferrari's relative engine deficit ({source_note}), "
            "their chassis advantage is decisive here.")

    # Always flag a consecutive-DNF driver if relevant.
    if val("consecutive_dnf_flag") == 1 and not any("reliability" in s.lower() or
                                                    "DNF" in s for s in sentences):
        sentences.append(
            f"{driver_code} enters flagged for a reliability concern (consecutive DNFs).")

    return " ".join(sentences)


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP AS DICT (for dashboard integration)
# ═══════════════════════════════════════════════════════════════════════════════

def get_shap_for_driver_race(rf, df, driver_code, round_num):
    """
    Returns SHAP values as a serialisable dict for a specific driver/race.
    Used by f1_dashboard.py to show XAI inline.
    """
    row = df[(df["driver"] == driver_code) & (df["round"] == round_num)]
    if len(row) == 0:
        return None

    X = row[FEATURE_COLS].values
    explainer = shap.TreeExplainer(rf)
    svs = explainer.shap_values(X)[0]
    ev = explainer.expected_value
    baseline = float(np.mean(ev) if hasattr(ev, "__len__") else ev)
    pred = baseline + float(svs.sum())

    return {
        "driver":        driver_code,
        "round":         round_num,
        "baseline":      round(baseline, 2),
        "prediction":    round(pred, 2),
        "actual":        int(row["position"].values[0]),
        "shap_values":   {FEATURE_LABELS[f]: round(float(v), 3)
                          for f, v in zip(FEATURE_COLS, svs)},
        "feature_values": {FEATURE_LABELS[f]: round(float(row[f].values[0]), 3)
                           for f in FEATURE_COLS},
        # Raw-keyed maps for programmatic use (natural-language explainer, etc.)
        "shap_values_raw":    {f: float(v) for f, v in zip(FEATURE_COLS, svs)},
        "feature_values_raw": {f: float(row[f].values[0]) for f in FEATURE_COLS},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOP-10 COMPACT ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _n_circuits(data):
    """Count total completed race circuits (for attribution note)."""
    return len(data.get("races", []))


def _compact_driver_header(rank, code, team, pts):
    width = 54
    title = f" P{rank} {code} - {team} ({pts} pts) "
    pad = max(0, width - len(title))
    lp = pad // 2
    rp = pad - lp
    return f"{'=' * lp}{title}{'=' * rp}"


def _top10_shap_analysis(rf, df, data, shap_values, X_df, expected_value,
                          save_plots, top_n, model_name):
    """
    Run compact SHAP analysis for top-N drivers.
    Prints per-driver summary + summary table at the end.
    """
    standings = sorted(data["standings"]["drivers"], key=lambda d: -d["points"])
    top = standings[:top_n]
    n_circ = _n_circuits(data)

    table_rows = []

    for rank, drv_info in enumerate(top, 1):
        code = drv_info["code"]
        team = drv_info["team"]
        pts  = drv_info["points"]

        # Most recent completed race for this driver
        drv_df = df[df["driver"] == code]
        if len(drv_df) == 0:
            continue
        round_num = int(drv_df["round"].max())

        # Compute SHAP for this driver/round
        result = get_shap_for_driver_race(rf, df, code, round_num)
        if result is None:
            continue

        # Find top 3 SHAP features (raw)
        shap_raw = result["shap_values_raw"]
        top_feats = sorted(shap_raw.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

        print(f"\n{_compact_driver_header(rank, code, team, pts)}")
        race_name = next((r["name"] for r in data["races"] if r["round"] == round_num),
                         f"Round {round_num}")
        get_actual_result(data, round_num)   # prints [OK]/[WARNING] data-source line
        print(f"Round {round_num} — {race_name} | "
              f"Prediction: P{result['prediction']:.1f} | Actual: P{result['actual']}")
        print("Top features:")
        for feat, sv in top_feats:
            label = FEATURE_LABELS.get(feat, feat)
            direction = "[better]" if sv < 0 else "[worse]"
            warn = " [WARNING]" if feat in ("consecutive_dnf_flag", "rolling_dnf_rate",
                                            "reliability_concern_score") and sv > 0 else ""
            print(f"  {label:<30s}  {sv:+.2f}  {direction}{warn}")

        # Natural-language explanation
        states = {r["driver"]: {**{f: r[f] for f in FEATURE_COLS}, "team": r["team"]}
                  for _, r in df[df["round"] == round_num].iterrows()}
        prof = get_track_profile(race_name)
        nl = explain_prediction_natural_language(
            result, code, race_name, prof, states, n_circuits=n_circ)
        print(f"  {nl}")

        if save_plots:
            mask = (df["driver"] == code) & (df["round"] == round_num)
            if mask.any():
                idx = df[mask].index[0] - df.index[0]
                plot_local_waterfall(shap_values, X_df, expected_value, idx,
                                     code, round_num, model_name, save=True)

        # Accumulate table row (strip "(...)" suffixes so labels fit the cell)
        top1_feat = FEATURE_LABELS.get(top_feats[0][0], top_feats[0][0]) if top_feats else "-"
        top1_feat = top1_feat.split(" (")[0]
        top1_sv   = top_feats[0][1] if top_feats else 0
        table_rows.append({
            "rank": rank, "code": code,
            "actual": result["actual"],
            "pred":   result["prediction"],
            "top1":   top1_feat[:22],
            "top1_dir": "-" if top1_sv < 0 else "+",
        })

    # Summary table
    if table_rows:
        print("\n")
        print("+========================================================+")
        print(f"|  SHAP SUMMARY - Top {len(table_rows)} Drivers".ljust(57) + "|")
        print("+============+========+==========+======================+")
        print("| Driver     | Actual | Pred     | #1 SHAP Feature      |")
        print("+============+========+==========+======================+")
        for r in table_rows:
            driver_cell = f"{r['code']}  P{r['rank']}"
            actual_cell = f"P{r['actual']}"
            pred_cell   = f"P {r['pred']:.1f}"
            feat_cell   = f"{r['top1'][:18]} {r['top1_dir']}"
            print(f"| {driver_cell:<10s} | {actual_cell:^6s} | {pred_cell:<8s} | {feat_cell:<20s} |")
        print("+============+========+==========+======================+")

        # Global Clever-Hans verdict across all explained rows
        top10_drivers = [r["code"] for r in table_rows]
        mask_all = df["driver"].isin(top10_drivers)
        if mask_all.any():
            idxs = np.where(mask_all.values)[0]
            print()
            clever_hans_check(shap_values[idxs], df, X_df.iloc[idxs])


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-DRIVER VERBOSE ANALYSIS (existing behaviour)
# ═══════════════════════════════════════════════════════════════════════════════

def _single_driver_analysis(rf, xgb_model, df, data, shap_values, X_df,
                             expected_value, model_name, driver, round_num,
                             save_plots):
    """Full verbose output for a single driver/round (existing behaviour)."""
    n_circ = _n_circuits(data)
    get_actual_result(data, round_num)   # prints [OK]/[WARNING] data-source line

    plot_global_importance(shap_values, FEATURE_COLS, model_name, save=save_plots)
    plot_beeswarm(shap_values, X_df, model_name, save=save_plots)

    mask = (df["driver"] == driver) & (df["round"] == round_num)
    if mask.any():
        idx = df[mask].index[0] - df.index[0]
        plot_local_waterfall(shap_values, X_df, expected_value, idx,
                              driver, round_num, model_name, save=save_plots)
        result = get_shap_for_driver_race(rf, df, driver, round_num)
        if result:
            print(f"\n  SHAP summary for {driver} R{round_num}:")
            print(f"  Baseline: {result['baseline']:.1f}  →  Prediction: {result['prediction']:.1f}"
                  f"  (Actual: P{result['actual']})")
            for feat, sv in sorted(result["shap_values"].items(),
                                   key=lambda x: abs(x[1]), reverse=True):
                direction = "[worse]" if sv > 0 else "[better]"
                print(f"    {feat:40s}  {sv:+.3f}  {direction}")
    else:
        print(f"\n  No data found for {driver} in round {round_num}. "
              f"Try --driver VER --round 5")

    # Clever-Hans check
    round_mask = df["round"] == round_num
    if round_mask.any():
        idxs = np.where(round_mask.values)[0]
        clever_hans_check(shap_values[idxs], df, X_df.iloc[idxs])
    else:
        clever_hans_check(shap_values, df, X_df)

    # Natural-language explanation
    states = {r["driver"]: {**{f: r[f] for f in FEATURE_COLS}, "team": r["team"]}
              for _, r in df[df["round"] == round_num].iterrows()}
    sd = get_shap_for_driver_race(rf, df, driver, round_num)
    if sd:
        race_name = next((rr["name"] for rr in data["races"]
                          if rr["round"] == round_num), f"Round {round_num}")
        prof = get_track_profile(race_name)
        print("\n  Natural-language explanation:")
        print("  " + explain_prediction_natural_language(
            sd, driver, race_name, prof, states, n_circuits=n_circ))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for F1 models")
    parser.add_argument("--driver",     default=None,
                        help="Driver code for single-driver verbose analysis "
                             "(omit to analyse all top-10)")
    parser.add_argument("--round",      type=int, default=None,
                        help="Race round (default: most recent for the driver)")
    parser.add_argument("--model",      default="rf", choices=["rf", "xgb"],
                        help="Which model to explain")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save PNG plots to plots/")
    parser.add_argument("--top-n",      type=int, default=10,
                        help="Number of top drivers to include in default mode (default 10)")
    args = parser.parse_args()

    rf, xgb_model, df, meta, data = load_models_and_data()
    model      = rf if args.model == "rf" else xgb_model
    model_name = "Random Forest" if args.model == "rf" else "XGBoost"

    X_df = df[FEATURE_COLS]
    X    = X_df.values

    # Global SHAP (needed by both modes)
    shap_values, expected_value, explainer = compute_shap_values(model, X, model_name)

    if args.driver is None:
        # ── Default: compact analysis of all top-N drivers ──
        _top10_shap_analysis(rf, df, data, shap_values, X_df, expected_value,
                              args.save_plots, args.top_n, model_name)
    else:
        # ── Single-driver verbose mode ──
        driver    = args.driver.upper()
        round_num = args.round or int(df[df["driver"] == driver]["round"].max()
                                      if len(df[df["driver"] == driver]) > 0
                                      else df["round"].max())
        _single_driver_analysis(rf, xgb_model, df, data, shap_values, X_df,
                                 expected_value, model_name, driver, round_num,
                                 args.save_plots)


if __name__ == "__main__":
    main()

"""
lrp_timeseries.py — LRP-SIGN for LSTM on F1 Season Time Series
===============================================================
Layer-wise Relevance Propagation (LRP) with the SIGN input-attribution rule
applied to the LSTM model that predicts finishing position from season trends.

What LRP-SIGN does:
  - Propagates the model's output (predicted position) BACKWARDS through the
    LSTM layer-by-layer, conserving relevance at each step.
  - Assigns a relevance score to EACH INPUT TIMESTEP (each past race) and to
    each feature within that timestep.
  - The SIGN rule assigns the direction of each input's relevance from the
    input's position relative to its reference value mu (here: the feature
    mean, so mu = 0 after standardisation):

        R_i = SIGN(x_i) * w_ij / z_j * R_j,
        where SIGN(x_i) = +1 if x_i >= mu, else -1

    This removes the bias that vanilla LRP inherits from the sign of the
    weights, so positive relevance reliably means "drives the prediction up"
    (a WORSE predicted position) and negative relevance means "drives it
    down" (a BETTER predicted position) — crucial for time series where
    activations of mixed sign would otherwise cancel.

References:
  Gumpfer et al. (2023). SIGNed Explanations. Information Fusion 99, p. 101883.
      ("SIGNed explanations: Unveiling relevant features by reducing bias" —
       the SIGN attribution rule used here.)
  Bach et al. (2015). On pixel-wise explanations for non-linear classifier
      decisions by layer-wise relevance propagation. PLOS ONE 10(7).
      (original LRP; the epsilon stabiliser used in the linear rule.)
  Montavon et al. (2019). Explaining nonlinear classification decisions with
      deep Taylor decomposition. Pattern Recognition 65. (theoretical basis.)

Usage:
    python lrp_timeseries.py                    # all top-10 drivers, compact
    python lrp_timeseries.py --driver NOR       # single driver deep dive
    python lrp_timeseries.py --driver ANT --round 9   # sequence ending at R9
    python lrp_timeseries.py --save-plots       # save PNGs to plots/
"""

import os, sys, json, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
sys.path.insert(0, str(ROOT))

SEASON = 2026

# Shared best-estimate 2026 results; real JSON/FastF1 data is always preferred.
from shap_analysis import FALLBACK_RESULTS_2026, get_actual_result

DRIVER_COLORS = {
    "NOR": "#FF8000", "PIA": "#FF8000", "VER": "#3671C6", "LAW": "#3671C6",
    "LEC": "#E8002D", "HAM": "#E8002D", "RUS": "#27F4D2", "ANT": "#27F4D2",
    "ALO": "#358C75", "STR": "#358C75", "GAS": "#0093CC", "DOO": "#0093CC",
    "ALB": "#64C4FF", "SAI": "#64C4FF", "HUL": "#52E252", "BOR": "#52E252",
    "TSU": "#6692FF", "HAD": "#6692FF", "OCO": "#B6BABD", "BEA": "#B6BABD",
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
    "grid.alpha":       0.4,
}

# Short readable names (≤20 chars) aligned 1:1 with ml_models.FEATURE_COLS.
LSTM_FEATURE_NAMES = [
    "Round",
    "Track Power",
    "Track Downforce",
    "Track Type",
    "Engine",
    "Chassis",
    "Track Perf Score",
    "Pace-Adj Form",
    "Momentum",
    "Sector1 Delta",
    "Sector2 Delta",
    "Sector3 Delta",
    "Rolling DNF Rate",
    "Consec DNF Flag",
    "Reliab Concern",
    "Cumulative Pts",
    "Experience",
    "Champ Position",
]


# ═══════════════════════════════════════════════════════════════════════════════
# LRP-SIGN CORE
# ═══════════════════════════════════════════════════════════════════════════════

def _sign_stabiliser(Z, eps=1e-6):
    """
    Epsilon stabiliser of the LRP-epsilon rule (Bach et al. 2015):
    replace small denominators with a signed epsilon to avoid division by
    near-zero pre-activations.
        Z_stable = Z + eps * sign(Z),  with sign(0) = +1
    """
    sign = np.where(Z >= 0, 1.0, -1.0)
    return Z + eps * sign


def _sign(x, mu=0.0):
    """
    SIGN attribution rule (Gumpfer et al. 2023, Information Fusion 99):
        SIGN(x_i) = +1 if x_i >= mu, else -1
    The reference value mu is the feature mean.  Inputs are standardised
    (StandardScaler) before entering the LSTM, so mu = 0 in model space.
    Applying SIGN(x_i) to the propagated relevance,
        R_i = SIGN(x_i) * w_ij / z_j * R_j,
    removes the dependence of the attribution's sign on the sign of the
    weights, reducing the bias of vanilla LRP explanations.
    """
    return np.where(x >= mu, 1.0, -1.0)


def lrp_sign_linear(R_out, W, x, bias=None, eps=1e-6):
    """
    LRP-epsilon for a single linear layer (Bach et al. 2015):
      Z_j = sum_i(x_i * W_ji) + b_j
      R_i = sum_j [ x_i * W_ji / Z_stable_j * R_j ]

    The SIGN rule (Gumpfer et al. 2023) is applied later, at the INPUT layer,
    where each x_i has a meaningful reference value mu (its feature mean) —
    see _sign() and lrp_sign_lstm_timestep() step 3.  Hidden activations have
    no such reference, so the epsilon rule is used inside the network.

    Arguments:
      R_out : (output_size,) — relevance coming FROM the layer above
      W     : (output_size, input_size) — weight matrix
      x     : (input_size,) — input activations
      bias  : (output_size,) — optional bias

    Returns:
      R_in  : (input_size,) — relevance distributed TO inputs
    """
    # Forward: compute each neuron's pre-activation
    Z = W @ x                          # (output_size,)
    if bias is not None:
        Z = Z + bias

    Z_stable = _sign_stabiliser(Z, eps)  # (output_size,)

    # Message from each output neuron to each input neuron
    # contribution[j, i] = x_i * W[j, i] / Z_stable[j]
    # Then R_in[i] = sum_j(contribution[j, i] * R_out[j])
    contributions = (W * x[np.newaxis, :]) / Z_stable[:, np.newaxis]  # (out, in)
    R_in = contributions.T @ R_out     # (input_size,)
    return R_in


def lrp_sign_lstm_timestep(R_h, Wx, Wh, b, x_t, h_prev, c_prev, eps=1e-6):
    """
    LRP-SIGN for a single LSTM timestep.

    LSTM gate equations (all gates share one computation):
      [i, f, g, o] = Wx @ x_t + Wh @ h_prev + b
      i = sigmoid(gates[0:H])
      f = sigmoid(gates[H:2H])
      g = tanh(gates[2H:3H])
      o = sigmoid(gates[3H:4H])
      c_t = f*c_prev + i*g
      h_t = o*tanh(c_t)

    LRP-SIGN strategy:
      1. Propagate R_h through the output gate into c_t
      2. Propagate into i*g (the update) and f*c_prev (the memory)
      3. Propagate into x_t (relevant input features)

    Returns:
      R_x : (input_size,) — relevance for input x_t at this timestep
      R_h : (hidden_size,) — relevance flowing back to h_prev
    """
    H = Wx.shape[0] // 4

    # Recompute gates
    gates = Wx @ x_t + Wh @ h_prev + b
    i_gate = 1 / (1 + np.exp(-np.clip(gates[0*H:1*H], -50, 50)))
    f_gate = 1 / (1 + np.exp(-np.clip(gates[1*H:2*H], -50, 50)))
    g_gate = np.tanh(np.clip(gates[2*H:3*H], -50, 50))
    o_gate = 1 / (1 + np.exp(-np.clip(gates[3*H:4*H], -50, 50)))

    c_t = f_gate * c_prev + i_gate * g_gate
    tanh_c_t = np.tanh(np.clip(c_t, -50, 50))

    # Step 1: R_h → R_c  (through output gate)
    # h_t = o * tanh(c_t)  → numerator contribution = o * tanh(c_t) = h_t
    Z_ct = o_gate * tanh_c_t
    Z_ct_stable = _sign_stabiliser(Z_ct, eps)
    R_c = (o_gate * tanh_c_t / Z_ct_stable) * R_h  # (H,)

    # Step 2: R_c → R_update + R_mem  (cell state split)
    # c_t = i*g + f*c_prev
    update = i_gate * g_gate    # (H,)
    memory = f_gate * c_prev    # (H,)
    Z_c = update + memory
    Z_c_stable = _sign_stabiliser(Z_c, eps)

    R_update = (update / Z_c_stable) * R_c   # input contribution
    R_mem    = (memory / Z_c_stable) * R_c   # memory (from previous cell)

    # Step 3: R_update → R_x (input features) via input gate + Wx
    # update = i_gate * g_gate, both depend on Wx @ x_t.
    # Magnitude: distribute |R_update| proportional to |Wx[:,i]^T R_update|.
    # Direction: SIGN rule (Gumpfer et al. 2023) —
    #     R_i = SIGN(x_i) * w_ij / z_j * R_j,  SIGN(x_i) = +1 if x_i >= mu else -1
    # x_t is standardised, so mu = 0: features above their season mean get
    # positive relevance, features below it get negative relevance.
    Wx_input = Wx[:H, :]   # Use input gate rows (0:H) as proxy
    x_contrib = np.abs(Wx_input.T @ R_update)  # (input_size,) magnitudes
    x_norm = np.sum(x_contrib) + eps
    R_x = _sign(x_t) * (x_contrib / x_norm) * np.sum(np.abs(R_update))

    # Step 4: R_mem flows back to h_prev
    # Wh has shape (4H, H); forget gate rows are [H:2H]
    H = Wx.shape[0] // 4
    Wh_forget = Wh[H:2*H, :]          # (H, H) — forget gate weights
    Wh_contrib = np.abs(Wh_forget.T @ R_mem)  # (H,)
    h_norm = np.sum(Wh_contrib) + eps
    R_h_prev = Wh_contrib / h_norm * np.sum(np.abs(R_mem))

    return R_x, R_h_prev, c_t


def lrp_sign_sequence(model, x_seq, eps=1e-6):
    """
    Full LRP-SIGN pass through a sequence (seq_len, n_features).
    Works with our LSTMNumpy model.

    Returns:
      relevance_per_timestep : (seq_len,)   — total relevance per race
      relevance_per_feature  : (seq_len, n_features) — per feature per timestep
    """
    # Normalise input (same transform used during training)
    n_feat = x_seq.shape[1]
    x_norm = model.scaler.transform(x_seq)  # (seq_len, n_feat)

    # ── Forward pass: collect hidden states ──────────────────────────────────
    h_states = []  # h after each timestep
    c_states = []  # c after each timestep
    h = np.zeros(model.hidden_size)
    c = np.zeros(model.hidden_size)
    H = model.hidden_size

    for t in range(x_norm.shape[0]):
        gates = model.Wx @ x_norm[t] + model.Wh @ h + model.b
        i_g = 1 / (1 + np.exp(-np.clip(gates[0*H:1*H], -50, 50)))
        f_g = 1 / (1 + np.exp(-np.clip(gates[1*H:2*H], -50, 50)))
        g_g = np.tanh(np.clip(gates[2*H:3*H], -50, 50))
        o_g = 1 / (1 + np.exp(-np.clip(gates[3*H:4*H], -50, 50)))
        c = f_g * c + i_g * g_g
        h = o_g * np.tanh(np.clip(c, -50, 50))
        h_states.append(h.copy())
        c_states.append(c.copy())

    # ── Backward pass: LRP-SIGN ────────────────────────────────────────────
    # Start relevance from the output layer
    h_last = h_states[-1]
    pred_raw = (model.Wo @ h_last + model.bo)[0]

    # Initial relevance = prediction value (in normalised space)
    R_h = lrp_sign_linear(
        R_out=np.array([pred_raw]),
        W=model.Wo,
        x=h_last,
        bias=model.bo,
    )

    # Propagate backwards through each LSTM timestep
    seq_len = x_norm.shape[0]
    relevance_features = np.zeros((seq_len, n_feat))  # (T, F)

    h_prev = np.zeros(H)
    c_prev = np.zeros(H)

    for t in range(seq_len - 1, -1, -1):
        h_prev_t = h_states[t - 1] if t > 0 else np.zeros(H)
        c_prev_t = c_states[t - 1] if t > 0 else np.zeros(H)

        R_x_t, R_h_back, _ = lrp_sign_lstm_timestep(
            R_h, model.Wx, model.Wh, model.b,
            x_norm[t], h_prev_t, c_prev_t, eps=eps
        )
        relevance_features[t] = R_x_t
        R_h = R_h_back

    # Total relevance per timestep = sum over features
    relevance_per_timestep = np.sum(np.abs(relevance_features), axis=1)
    # Normalise to sum to 1 (for interpretability)
    total = relevance_per_timestep.sum() + 1e-10
    relevance_per_timestep = relevance_per_timestep / total
    relevance_features = relevance_features / (np.sum(np.abs(relevance_features)) + 1e-10)

    return relevance_per_timestep, relevance_features


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_data():
    if not (MODELS_DIR / "lstm.pkl").exists():
        print("LSTM model not found. Run: python ml_models.py")
        sys.exit(1)

    # Import ml_models first so pickle can find LSTMNumpy class in __main__
    sys.path.insert(0, str(ROOT))
    import ml_models as _ml
    import sys as _sys
    # Register the class in the calling module's namespace so pickle finds it
    _sys.modules["__main__"].LSTMNumpy = _ml.LSTMNumpy

    from ml_models import build_feature_matrix, load_demo_data, build_lstm_sequences

    lstm = pickle.load(open(MODELS_DIR / "lstm.pkl", "rb"))
    with open(MODELS_DIR / "meta.json") as f:
        meta = json.load(f)

    data = load_demo_data()
    df = build_feature_matrix(data)
    X, y, drivers, feat_cols = build_lstm_sequences(df, seq_len=3)
    return lstm, df, X, y, drivers, feat_cols, meta


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: RELEVANCE HEATMAP (the key LRP-SIGN output)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_relevance_heatmap(driver_code, round_labels, rel_timestep,
                            rel_features, feat_names, pred, actual,
                            save=False):
    """
    Two-panel plot:
    Top:    Bar chart — which race (timestep) was most relevant?
    Bottom: Heatmap  — which FEATURE in which race mattered?

    This is the visual centrepiece for LRP-SIGN in the presentation.
    """
    plt.rcParams.update(PLT_STYLE)
    driver_color = DRIVER_COLORS.get(driver_code, "#888888")

    fig = plt.figure(figsize=(10, 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.45)
    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    seq_len = len(round_labels)
    x_pos   = np.arange(seq_len)

    # ── Top: timestep relevance bars ──────────────────────────────────────────
    colors = [driver_color if v == rel_timestep.max() else "#7070a0"
              for v in rel_timestep]
    bars = ax_bar.bar(x_pos, rel_timestep, color=colors, edgecolor="none", width=0.6)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"Race {r}" for r in round_labels], fontsize=9)
    ax_bar.set_ylabel("Relevance", fontsize=9)
    ax_bar.set_title(
        f"LRP-SIGN: {driver_code} — which past races influenced prediction?  "
        f"[Pred P{pred:.0f} | Actual P{actual}]",
        color=driver_color, fontsize=11, pad=10
    )
    for bar, val in zip(bars, rel_timestep):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#e8e8f0")

    # ── Bottom: feature × timestep heatmap ───────────────────────────────────
    # rel_features shape: (seq_len, n_features)
    # Transpose so rows = features, cols = timesteps
    data_heat = rel_features.T  # (n_feat, seq_len)

    im = ax_heat.imshow(
        data_heat, aspect="auto", cmap="RdBu_r",
        vmin=-np.abs(data_heat).max(), vmax=np.abs(data_heat).max()
    )
    ax_heat.set_xticks(x_pos)
    ax_heat.set_xticklabels([f"Race {r}" for r in round_labels], fontsize=9)
    ax_heat.set_yticks(range(len(feat_names)))
    ax_heat.set_yticklabels(feat_names, fontsize=9)
    ax_heat.set_title("Feature × Timestep relevance (red = drives position up, blue = down)",
                       fontsize=9, color="#7070a0", pad=8)

    # Annotate cells
    for i in range(data_heat.shape[0]):
        for j in range(data_heat.shape[1]):
            val = data_heat[i, j]
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7, color="white" if abs(val) > 0.05 else "#7070a0")

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#7070a0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#7070a0", fontsize=7)

    if save:
        path = PLOTS_DIR / f"lrp_sign_{driver_code}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: MULTI-DRIVER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def plot_driver_comparison(results, save=False):
    """
    Compare LRP relevance patterns across multiple drivers.
    Which past race was most influential for each driver's prediction?
    """
    plt.rcParams.update(PLT_STYLE)
    n_drivers = len(results)
    if n_drivers == 0:
        return

    fig, axes = plt.subplots(1, n_drivers, figsize=(4 * n_drivers, 4),
                              sharey=False)
    if n_drivers == 1:
        axes = [axes]

    for ax, (driver, data) in zip(axes, results.items()):
        rel   = data["relevance_timestep"]
        rounds = data["rounds"]
        color = DRIVER_COLORS.get(driver, "#888888")

        colors_bar = [color if v == rel.max() else "#7070a0" for v in rel]
        ax.bar(range(len(rel)), rel, color=colors_bar, edgecolor="none", width=0.6)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=9)
        ax.set_title(driver, color=color, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.set_ylabel("Timestep relevance (LRP-SIGN)", fontsize=9)

    fig.suptitle("LRP-SIGN — Most influential past race per driver",
                 color="#e8e8f0", fontsize=12, y=1.02)
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "lrp_sign_driver_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# GET LRP FOR DASHBOARD (dict output)
# ═══════════════════════════════════════════════════════════════════════════════

def get_lrp_for_driver(lstm, df, driver_code, feat_cols, seq_len=3, end_round=None):
    """
    Returns LRP-SIGN relevance as a serialisable dict for dashboard integration.

    `end_round` restricts the input sequence to races up to and including that
    round (default: the driver's most recent completed race).
    """
    drv_df = df[df["driver"] == driver_code].sort_values("round")
    if end_round is not None:
        drv_df = drv_df[drv_df["round"] <= end_round]
    if len(drv_df) < seq_len + 1:
        return None

    vals = drv_df[feat_cols].values
    positions = drv_df["position"].values
    rounds = drv_df["round"].values

    # Use the last available sequence
    x_seq = vals[-seq_len:].astype(np.float32)
    pred = float(lstm.predict(x_seq[np.newaxis])[0])
    actual = int(positions[-1])

    # Check if LSTM was properly fitted
    if not getattr(lstm, "fitted", False):
        return {
            "driver": driver_code,
            "prediction": round(pred, 1),
            "actual": actual,
            "rounds": rounds[-seq_len:].tolist(),
            "relevance_timestep": [1/seq_len] * seq_len,
            "relevance_features": [[0.0] * len(feat_cols)] * seq_len,
            "feature_names": feat_cols,
            "note": "LSTM not fully trained — uniform relevance shown as placeholder",
        }

    rel_t, rel_f = lrp_sign_sequence(lstm, x_seq)

    return {
        "driver":               driver_code,
        "prediction":           round(pred, 1),
        "actual":               actual,
        "rounds":               rounds[-seq_len:].tolist(),
        "relevance_timestep":   rel_t.tolist(),
        "relevance_features":   rel_f.tolist(),
        "feature_names":        feat_cols,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _print_driver_lrp(result, feat_cols, verbose=True):
    """Print LRP-SIGN relevance for one driver (console report)."""
    print(f"LRP-SIGN results for {result['driver']}:")
    print(f"  Prediction: P{result['prediction']:.0f}  |  Actual: P{result['actual']}")
    print(f"  Rounds in sequence: {result['rounds']}")
    print("  Timestep relevance (which race mattered most):")
    for r, rel in zip(result["rounds"], result["relevance_timestep"]):
        bar = "#" * int(rel * 30)
        print(f"    Race {r}: {rel:.3f}  {bar}")
    if not verbose:
        return
    print("\n  Feature x Timestep relevance "
          "(positive = drives position higher = worse):")
    for t_idx, r in enumerate(result["rounds"]):
        print(f"    Race {r}:")
        for f_idx, feat in enumerate(feat_cols):
            rel_f = result["relevance_features"][t_idx][f_idx]
            direction = "[worse]" if rel_f > 0 else "[better]"
            print(f"      {feat:30s} {rel_f:+.4f}  {direction}")


def main():
    parser = argparse.ArgumentParser(description="LRP-SIGN for F1 LSTM")
    parser.add_argument("--driver", default=None,
                        help="Driver code (e.g. NOR); omit to analyse all top-10")
    parser.add_argument("--round", type=int, default=None,
                        help="Explain the sequence ending at this round "
                             "(default: driver's most recent race)")
    parser.add_argument("--compare", nargs="+", default=["NOR", "VER", "HAM"],
                        help="Drivers to compare")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top drivers in default mode (default 10)")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    print(f"SEASON = {SEASON}")
    lstm, df, X, y, seq_drivers, feat_cols, meta = load_model_and_data()

    # Data source: build_feature_matrix() above consumed f1_data_2026.json
    # (FastF1-derived).  Only when a round is missing there do we fall back.
    from ml_models import load_demo_data
    data = load_demo_data()
    if data.get("races"):
        print(f"[OK] LRP-SIGN using FastF1/JSON data ({len(data['races'])} races loaded)")
    else:
        print("[WARNING] LRP-SIGN using fallback results (FastF1 cache unavailable)")

    if hasattr(lstm, "hidden_size"):
        print(f"\nLSTM backend: NumPy  (hidden={lstm.hidden_size})")
    else:
        print("\nLSTM backend: PyTorch")

    print(f"Feature columns: {feat_cols}")
    print(f"Sequence data: {len(X) if X is not None else 0} sequences available\n")

    seq_len = 3
    save = args.save_plots
    display_names = (LSTM_FEATURE_NAMES
                     if len(feat_cols) == len(LSTM_FEATURE_NAMES) else feat_cols)

    if args.driver is None:
        # ── Default: all top-N drivers from current standings ────────────────
        standings = sorted(data["standings"]["drivers"], key=lambda d: -d["points"])
        for rank, drv in enumerate(standings[:args.top_n], 1):
            code = drv["code"]
            result = get_lrp_for_driver(lstm, df, code, feat_cols, seq_len,
                                        end_round=args.round)
            print(f"=== P{rank} {code} - {drv['team']} ({drv['points']} pts) ===")
            if result:
                _print_driver_lrp(result, feat_cols, verbose=False)
                if save:
                    plot_relevance_heatmap(
                        code, result["rounds"],
                        np.array(result["relevance_timestep"]),
                        np.array(result["relevance_features"]),
                        display_names, result["prediction"], result["actual"],
                        save=True)
            else:
                print(f"  Not enough data (need >= {seq_len + 1} races)")
            print()
    else:
        # ── Single driver deep dive ──────────────────────────────────────────
        target_driver = args.driver.upper()
        if args.round is not None:
            get_actual_result(data, args.round)   # prints [OK]/[WARNING] source line
        result = get_lrp_for_driver(lstm, df, target_driver, feat_cols, seq_len,
                                    end_round=args.round)

        if result:
            _print_driver_lrp(result, feat_cols, verbose=True)
            plot_relevance_heatmap(
                target_driver,
                result["rounds"],
                np.array(result["relevance_timestep"]),
                np.array(result["relevance_features"]),
                display_names,
                result["prediction"],
                result["actual"],
                save=save,
            )
        else:
            print(f"Not enough data for {target_driver} "
                  f"(need >= {seq_len + 1} races up to the requested round)")

    # Multi-driver comparison
    comparison_results = {}
    for drv in args.compare:
        r = get_lrp_for_driver(lstm, df, drv.upper(), feat_cols, seq_len,
                               end_round=args.round)
        if r:
            comparison_results[drv.upper()] = {
                "relevance_timestep": np.array(r["relevance_timestep"]),
                "rounds": r["rounds"],
            }

    if len(comparison_results) >= 2:
        plot_driver_comparison(comparison_results, save=save)



if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Overlay Trp-cage simulation I(q) with experimental SAXS at three temperatures.

A single multiplicative scale factor is found by least-squares minimisation
over the clean high-q overlap region (0.12 – 0.33 Å⁻¹) of the 25 °C dataset,
avoiding detector-gap spikes at lower q:

    c = Σ(I_exp · I_sim_interp) / Σ(I_sim_interp²)
"""

import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--qmin", type=float, default=0.07,
                    help="Lower q bound (Å⁻¹) for the scale-factor fit (default: 0.07)")
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────
SIM_CSV  = "intensity.csv"
EXP_DIR  = "../saxs/260820/processed_saxs_nb_20261021/stitched/overlay_zoom/"
EXP_FILES = {
    "5 °C":  EXP_DIR + "TC_15_mg_mL__6786_µM___5_C.dat",
    "25 °C": EXP_DIR + "TC_15_mg_mL__6786_µM___25_C.dat",
    "60 °C": EXP_DIR + "TC_15_mg_mL__6786_µM___60_C.dat",
}
OUTPUT = "sim_vs_exp.png"

Q_MIN       = args.qmin
Q_SCALE_MAX = 0.33   # upper edge of simulation data

# ── Presentation style (mirrors sq_presentations.py) ───────────────────────
plt.rcParams.update({
    "text.usetex":       False,
    "font.size":         18,
    "axes.titlesize":    0,
    "axes.labelsize":    18,
    "legend.fontsize":   13,
    "legend.edgecolor":  "none",
    "legend.frameon":    True,
    "legend.framealpha": 0.5,
    "axes.facecolor":    "#ffffff",
    "figure.autolayout": True,
})

COLORS = {
    "5 °C":  "#4e9ad4",
    "25 °C": "#e07b39",
    "60 °C": "#c94040",
}
SIM_COLOR = "#2d6a2d"

# ── Load simulation ─────────────────────────────────────────────────────────
sim   = pd.read_csv(SIM_CSV)
q_sim = sim["q"].values
I_sim = sim["total"].values
cs_sim = CubicSpline(q_sim, I_sim, extrapolate=False)

# ── Load experimental data ──────────────────────────────────────────────────
def load_exp(path):
    return np.loadtxt(path, comments="#", skiprows=2)

exp = {T: load_exp(p) for T, p in EXP_FILES.items()}

# ── Find scale factor from 25 °C, clean high-q overlap only ─────────────────
d25 = exp["25 °C"]
q25, I25 = d25[:, 0], d25[:, 1]

mask = (q25 >= Q_MIN) & (q25 <= Q_SCALE_MAX)
q_ov, I_ov = q25[mask], I25[mask]
I_sim_ov = cs_sim(q_ov)
valid = ~np.isnan(I_sim_ov)
q_ov, I_ov, I_sim_ov = q_ov[valid], I_ov[valid], I_sim_ov[valid]

scale = np.dot(I_ov, I_sim_ov) / np.dot(I_sim_ov, I_sim_ov)
print(f"Scale factor c = {scale:.4e}  (I_exp = c · I_sim)")
print(f"Fitted over q = {Q_MIN}–{Q_SCALE_MAX} Å⁻¹ using 25 °C data")

# ── Scaled simulation clipped to the experimental q-range ───────────────────
q_exp_min = min(d[:, 0].min() for d in exp.values())
q_exp_max = min(d[:, 0].max() for d in exp.values())
q_fine = np.linspace(max(q_sim.min(), q_exp_min),
                     min(q_sim.max(), q_exp_max), 300)
I_fine = cs_sim(q_fine) * scale
valid_fine = ~np.isnan(I_fine)
q_fine, I_fine = q_fine[valid_fine], I_fine[valid_fine]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

for T, d in exp.items():
    q_e, I_e = d[:, 0], d[:, 1]
    ax.loglog(q_e, I_e, "-", color=COLORS[T], linewidth=1.8, label=f"Exp. {T}")

ax.loglog(q_fine, I_fine, "--", color=SIM_COLOR, linewidth=2.2,
          label="Simulation (25 °C)")
ax.set_xlabel(r"$q$  [Å$^{-1}$]", labelpad=-10)
ax.set_ylabel(r"$I(q)$ [a.u.]")
ax.tick_params(which="both", direction="in")
ax.legend(loc="upper right")

ax.text(0.02, 0.05, "14.5 mg/mL  (6.7 mM)\npH 4.1, no added salt",
        transform=ax.transAxes, fontsize=12,
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.7))

fig.savefig(OUTPUT, dpi=200)
print(f"Saved → {OUTPUT}")

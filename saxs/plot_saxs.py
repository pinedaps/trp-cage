#!/usr/bin/env python3
"""
saxs_plot.py — SAXS I(q) and Sig(q) plotter.

Supports any number of q-range groups (detector distances) in a single folder.
For each folder in --data the script creates:

  <folder>/plots/
    overlay/          one figure per q-range group, all samples overlaid
    individuals/      one figure per sample per q-range
    overlap/
      overlay/        all sample series with all ranges on the same axes
      individuals/    per-sample figure with all ranges overlaid
    stitched/
      overlay/        all chain-stitched curves overlaid
      individuals/    per-sample stitched curve + diagnostic

Usage
-----
    python plot_saxs.py --data 260813,260820 --base-dir saxs/
    python plot_saxs.py --data Preliminary_data_260721 --base-dir saxs/ --qcut 0.05
    python plot_saxs.py --data 260813 --base-dir saxs/ --normalize-exposure

Notes
-----
SAXS .dat files are already flux/monitor-normalised by the beamline reduction
pipeline, so additional exposure-time normalisation is off by default.
Use --normalize-exposure to opt in.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ── Constants ─────────────────────────────────────────────────────────────────
MW_DA         = 2210.5   # g/mol  (Trp-cage)
_SKIP_SUF     = ('_sub', '_merged', '_azimAvg', '_azimavg', '_azimaVg')
_DIST_CLUSTER = 1.15     # two distances are the same range if ratio < this
_PLOT_SNR     = False    # set True via --snr

# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'sans-serif'})
LW     = 0.75
DPI    = 150
A_GRID = 0.3
FS_LBL = 13
FS_TK  = 12
FS_LEG = 9
FS_TTL = 12


# ── Unit helpers ──────────────────────────────────────────────────────────────

def mg_to_uM(mg_mL: float) -> float:
    return mg_mL / MW_DA * 1e6


# ── Header parsing ────────────────────────────────────────────────────────────

def parse_header(filepath: Path) -> tuple[dict, list[str]]:
    meta: dict[str, str] = {}
    data_lines: list[str] = []
    n_sep = 0
    with open(filepath, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith('####'):
                n_sep += 1
                continue
            if n_sep < 2:
                m = re.match(r'#\s+(\S+)\s+(.*)', line)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
            else:
                if stripped and not stripped.startswith('q('):
                    data_lines.append(stripped)
    return meta, data_lines


def parse_comment(raw: str) -> dict:
    """
    Decode Comment field → structured dict.

    Fields: position (well code), type, label, conc_mg, conc_uM.
    """
    raw = raw.strip().rstrip('-').strip()
    m = re.match(r'^([A-Za-z]?\d+)\(([^)]+)\)', raw)
    if m:
        position = m.group(1)
        inner    = m.group(2).strip()
    else:
        position = ''
        inner    = raw

    lo = inner.lower()

    if re.search(r'\bmq\b|mqw|\bmilli.?q\b|\bmw\b', lo):
        return dict(position=position, type='mq', label='MQ water',
                    conc_mg=None, conc_uM=None)
    if 'buffer' in lo:
        return dict(position=position, type='buffer', label='buffer',
                    conc_mg=None, conc_uM=None)

    m_per = re.match(r'(?:tc\s*)?per(\d*)', lo)
    if m_per:
        num = m_per.group(1)
        return dict(position=position, type='per',
                    label=f'permeate {num}'.rstrip(),
                    conc_mg=None, conc_uM=None)

    m_ret = re.match(r'(?:tc\s*)?ret(\d*)', lo)
    if m_ret:
        num = m_ret.group(1)
        return dict(position=position, type='ret',
                    label=f'retentate {num}'.rstrip(),
                    conc_mg=None, conc_uM=None)

    m_tc = re.search(r'tc[\s_]*((\d+)p(\d+))', inner, re.IGNORECASE)
    if m_tc:
        conc_mg = float(m_tc.group(1).replace('p', '.'))
        conc_uM = mg_to_uM(conc_mg)
        return dict(position=position, type='tc',
                    label=f'TC {conc_mg:g} mg/mL ({conc_uM:.0f} µM)',
                    conc_mg=conc_mg, conc_uM=conc_uM)

    return dict(position=position, type='unknown', label=inner,
                conc_mg=None, conc_uM=None)


def make_label(meta: dict) -> tuple[str, dict]:
    sample = parse_comment(meta.get('Comment', ''))
    try:
        temp    = float(meta.get('Temperature-TC', meta.get('Temperature', '')))
        t_str   = f'{temp:.0f}°C'
    except (ValueError, TypeError):
        t_str   = '20°C'
    return f'{sample["label"]}, {t_str}', sample


def load_saxs(path: Path) -> dict:
    meta, data_lines = parse_header(path)
    q, I, sig = [], [], []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3:
            try:
                q.append(float(parts[0]))
                I.append(float(parts[1]))
                sig.append(float(parts[2]))
            except ValueError:
                continue
    label, sample = make_label(meta)
    return dict(
        q=np.array(q), I=np.array(I), sig=np.array(sig),
        meta=meta, label=label, sample=sample,
        exposure=float(meta.get('ExposureTime', 1.0)),
        dist=float(meta.get('SampleDistance', 0.0)),
        qrange='',          # filled in by cluster_qranges
        path=path, name=path.stem,
    )


def _skip(name: str) -> bool:
    return any(s in name for s in _SKIP_SUF)


# ── Q-range clustering ────────────────────────────────────────────────────────

def cluster_qranges(samples: list) -> dict[str, list]:
    """
    Group samples by SampleDistance using ratio-based clustering.

    range_1 = largest distance = lowest q
    range_N = smallest distance = highest q

    Assigns s['qrange'] in place.  Returns ordered dict {label: [samples]}.
    """
    if not samples:
        return {}

    dists = sorted({s['dist'] for s in samples})

    # Merge distances within _DIST_CLUSTER ratio
    clusters: list[list[float]] = [[dists[0]]]
    for d in dists[1:]:
        if d / clusters[-1][-1] < _DIST_CLUSTER:
            clusters[-1].append(d)
        else:
            clusters.append([d])

    # Sort clusters largest distance first (= lowest q first)
    clusters.sort(key=lambda c: -np.mean(c))

    groups: dict[str, list] = {}
    for i, cl in enumerate(clusters):
        label = f'range_{i + 1}'
        lo_d  = min(cl) * 0.9
        hi_d  = max(cl) * 1.1
        groups[label] = []
        for s in samples:
            if lo_d <= s['dist'] <= hi_d:
                s['qrange'] = label
                groups[label].append(s)

    n = len(clusters)
    if n == 2:
        # Rename for clarity when there are exactly two ranges
        groups['low_q']  = groups.pop('range_1')
        groups['high_q'] = groups.pop('range_2')
        for s in groups['low_q']:
            s['qrange'] = 'low_q'
        for s in groups['high_q']:
            s['qrange'] = 'high_q'

    return groups


# ── Blank subtraction ─────────────────────────────────────────────────────────

def assign_qranges_from_ref(samples: list, ref_range_groups: dict) -> None:
    """Assign qrange labels to samples by nearest mean distance in ref_range_groups."""
    cluster_means = {
        label: float(np.mean([s['dist'] for s in slist]))
        for label, slist in ref_range_groups.items()
    }
    for s in samples:
        closest = min(cluster_means, key=lambda l: abs(s['dist'] - cluster_means[l]))
        s['qrange'] = closest


def subtract_blank(samples: list, blank_samples: list) -> list:
    """
    Subtract blank from each sample; propagate uncertainties in quadrature.

    blank_samples must already have 'qrange' assigned matching the main data.
    Returns a new list (blank samples are excluded).
    """
    blank_by_range: dict[str, list] = {}
    for b in blank_samples:
        blank_by_range.setdefault(b['qrange'], []).append(b)

    blank_ids = {id(b) for b in blank_samples}
    result = []
    for s in samples:
        if id(s) in blank_ids:
            continue
        blanks = blank_by_range.get(s['qrange'], [])
        if not blanks:
            result.append(s)
            continue

        q_t = s['q']
        I_bl_list, sig2_bl_list = [], []
        for b in blanks:
            fi = interp1d(b['q'], b['I'],   kind='linear', bounds_error=False, fill_value=np.nan)
            fs = interp1d(b['q'], b['sig'], kind='linear', bounds_error=False, fill_value=np.nan)
            I_bl_list.append(fi(q_t))
            sig2_bl_list.append(fs(q_t) ** 2)

        I_bl   = np.nanmean(I_bl_list,   axis=0)
        sig_bl = np.sqrt(np.nanmean(sig2_bl_list, axis=0))

        s_new = dict(s)
        s_new['I']   = s['I']   - I_bl
        s_new['sig'] = np.sqrt(s['sig'] ** 2 + sig_bl ** 2)
        result.append(s_new)

    return result


# ── Output directory helpers ───────────────────────────────────────────────────

def make_plots_dir(folder: Path, suffix: str = '') -> Path:
    base = folder / f'plots{suffix}'
    if base.exists():
        ts   = datetime.now().strftime('%Y%m%d')
        base = folder / f'plots{suffix}_{ts}'
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Axis styling ──────────────────────────────────────────────────────────────

def _make_Isnr_fig(w, h):
    """Create figure with I(q) panel and, if _PLOT_SNR, a second SNR panel."""
    if _PLOT_SNR:
        fig, (ax_I, ax_snr) = plt.subplots(2, 1, figsize=(w, h), sharex=True, dpi=DPI)
    else:
        fig, ax_I = plt.subplots(1, 1, figsize=(w, h * 0.6), dpi=DPI)
        ax_snr = None
    return fig, ax_I, ax_snr


def _style_pair(ax_I, ax_snr, title=None):
    axes = [ax_I] if ax_snr is None else [ax_I, ax_snr]
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=A_GRID)
        ax.tick_params(labelsize=FS_TK)
    ax_I.set_ylabel(r'$I(q)$ [a.u.]', fontsize=FS_LBL)
    if ax_snr is not None:
        ax_snr.set_ylabel(r'$I(q)\,/\,\sigma(q)$', fontsize=FS_LBL)
        ax_snr.set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]', fontsize=FS_LBL)
        ax_snr.axhline(1, color='gray', lw=0.8, ls='--', alpha=0.7)
    else:
        ax_I.set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]', fontsize=FS_LBL)
    if title:
        ax_I.set_title(title, fontsize=FS_TTL)


def _valid(s: dict) -> np.ndarray:
    return (s['I'] > 0) & (s['sig'] > 0) & np.isfinite(s['I']) & np.isfinite(s['sig'])


def _colors(n: int) -> list:
    if n <= 10:
        return [plt.get_cmap('tab10')(i) for i in range(n)]
    return [plt.get_cmap('tab20')(i / max(n - 1, 1)) for i in range(n)]


def _safe(label: str) -> str:
    return re.sub(r'[^\w\-.]', '_', label)


# ── Individual and overlay plots ───────────────────────────────────────────────

def plot_individual(s: dict, outdir: Path):
    fig, ax_I, ax_snr = _make_Isnr_fig(6, 7)
    v = _valid(s)
    ax_I.plot(s['q'][v], s['I'][v], color='#1f77b4', lw=LW, label=s['label'])
    if ax_snr is not None:
        ax_snr.plot(s['q'][v], s['I'][v] / s['sig'][v], color='#1f77b4', lw=LW)
    _style_pair(ax_I, ax_snr, title=s['label'])
    ax_I.legend(fontsize=FS_LEG, frameon=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(outdir / f'{_safe(s["name"])}.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_overlay(samples: list, range_tag: str, outdir: Path):
    if not samples:
        return
    colors = _colors(len(samples))
    fig, ax_I, ax_snr = _make_Isnr_fig(7, 8)
    for s, c in zip(samples, colors):
        v = _valid(s)
        ax_I.plot(s['q'][v], s['I'][v], color=c, lw=LW, label=s['label'])
        if ax_snr is not None:
            ax_snr.plot(s['q'][v], s['I'][v] / s['sig'][v], color=c, lw=LW)
    _style_pair(ax_I, ax_snr, title=f'All samples — {range_tag}')
    ax_I.legend(fontsize=FS_LEG, frameon=True, loc='upper right')
    fig.tight_layout(pad=0.2)
    out = outdir / f'overlay_{range_tag}.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'    {out.relative_to(out.parents[3])}')


# ── Sample grouping by identity across ranges ──────────────────────────────────

def _identity_key(s: dict) -> str:
    """Canonical key for matching across q-ranges: position + bare sample label."""
    pos  = s['sample']['position']
    lbl  = s['sample']['label']          # no rep suffix, no temperature
    return f'{pos}|{lbl}' if pos else lbl


def _display_identity(ser: dict) -> str:
    """Human-readable identity label (drop the position prefix for display)."""
    ik = ser['identity']
    return ik.split('|', 1)[1] if '|' in ik else ik


def group_by_identity(range_groups: dict[str, list]) -> list[dict]:
    """
    Build a list of series: each series is one physical sample with
    measurements at one or more q-ranges.

    Returns list of dicts:
        identity  : str
        ranges    : {range_label: list_of_samples}  (sorted lowest-q first)
    """
    all_identities: dict[str, dict[str, list]] = {}
    range_order = list(range_groups.keys())   # already sorted low-q first

    for rlabel, samples in range_groups.items():
        for s in samples:
            ik = _identity_key(s)
            if ik not in all_identities:
                all_identities[ik] = {r: [] for r in range_order}
            all_identities[ik][rlabel].append(s)

    series = []
    for ik, r_dict in all_identities.items():
        series.append(dict(identity=ik, ranges=r_dict))
    return series


# ── Overlap plots ──────────────────────────────────────────────────────────────

_LS_CYCLE = ['-', '--', ':', '-.']

def plot_overlap_overlay(series_list: list, range_order: list, outdir: Path):
    if not series_list:
        return

    # Assign one color per physical sample
    n_identities = len(series_list)
    colors        = _colors(n_identities)

    fig, ax_I, ax_snr = _make_Isnr_fig(8, 9)

    for ser, c in zip(series_list, colors):
        # Take first available replicate per range
        for i, rlabel in enumerate(range_order):
            slist = ser['ranges'].get(rlabel, [])
            if not slist:
                continue
            s  = slist[0]
            ls = _LS_CYCLE[i % len(_LS_CYCLE)]
            v  = _valid(s)
            ax_I.plot(s['q'][v], s['I'][v], color=c, lw=LW, ls=ls,
                      label=s['label'] if i == 0 else '')
            if ax_snr is not None:
                ax_snr.plot(s['q'][v], s['I'][v] / s['sig'][v], color=c, lw=LW, ls=ls)

    # Line-style legend for ranges
    style_handles = [
        Line2D([0], [0], color='gray', lw=LW, ls=_LS_CYCLE[i % len(_LS_CYCLE)],
               label=rlabel.replace('_', '-'))
        for i, rlabel in enumerate(range_order)
    ]
    _style_pair(ax_I, ax_snr, title='Overlap — all samples')
    l1 = ax_I.legend(fontsize=FS_LEG, frameon=True, loc='upper right')
    ax_I.legend(handles=style_handles, fontsize=FS_LEG,
                loc='lower left', frameon=True)
    ax_I.add_artist(l1)
    fig.tight_layout(pad=0.2)
    out = outdir / 'overlap_overlay.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'    {out.relative_to(out.parents[4])}')


def plot_overlap_individual(ser: dict, range_order: list, outdir: Path):
    fig, ax_I, ax_snr = _make_Isnr_fig(7, 8)
    for i, rlabel in enumerate(range_order):
        slist = ser['ranges'].get(rlabel, [])
        for s in slist:
            v  = _valid(s)
            ls = _LS_CYCLE[i % len(_LS_CYCLE)]
            ax_I.plot(s['q'][v], s['I'][v], lw=LW, ls=ls,
                      label=f'{s["label"]} ({rlabel.replace("_","-")})')
            if ax_snr is not None:
                ax_snr.plot(s['q'][v], s['I'][v] / s['sig'][v], lw=LW, ls=ls)
    disp = _display_identity(ser)
    _style_pair(ax_I, ax_snr, title=f'Overlap — {disp}')
    ax_I.legend(fontsize=FS_LEG, frameon=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(outdir / f'{_safe(disp)}.png',
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ── Stitching ──────────────────────────────────────────────────────────────────

def _scale_factor(q_l, I_l, q_h, I_h, q_lo, q_hi) -> float:
    """Robust median log-ratio scale in the window [q_lo, q_hi]."""
    q_w = np.linspace(q_lo, q_hi, 200)
    fl  = interp1d(q_l, I_l, bounds_error=False, fill_value=np.nan)
    fh  = interp1d(q_h, I_h, bounds_error=False, fill_value=np.nan)
    Il  = fl(q_w)
    Ih  = fh(q_w)
    ok  = np.isfinite(Il) & np.isfinite(Ih) & (Il > 0) & (Ih > 0)
    if ok.sum() < 3:
        return 1.0
    return float(np.exp(np.median(np.log(Il[ok] / Ih[ok]))))


def find_stitch_point(lo: dict, hi: dict) -> tuple[float, float]:
    """
    Find the optimal stitch point in the overlap region.

    Algorithm
    ---------
    1. Compute overlap [q_ov_lo, q_ov_hi] and interpolate both branches onto
       a common log-spaced grid.
    2. Compute a *provisional* scale factor from the high-q 40 % of the overlap
       (top in log-q space).  The low-q edge of the high-q detector carries a
       beam-stop artefact (Ih >> Il) that biases a full-overlap median; the
       high-q portion is free of that contamination.
    3. Restrict the cut search to the central 60 % of the overlap in log-q
       space to avoid both the beam-stop edge and the low-q noise floor.
    4. Within the search window score every candidate q* by
           amplitude mismatch : |log(I_lo / I_hi_prov)|
         + derivative mismatch : 0.5 * |d(lnI)/d(lnq)_lo - …_hi| / (|slope_lo| + ε)
       and take the minimum as q_cut.
    5. Recompute the *final* scale from a local window (~10 % of overlap)
       centred on q_cut, guaranteeing continuity at the stitch point.

    References: Glatter & Kratky (1982); BioXTAS RAW (Hopkins et al. 2017).
    """
    q_l = lo['q'][(lo['I'] > 0) & np.isfinite(lo['I'])]
    I_l = lo['I'][(lo['I'] > 0) & np.isfinite(lo['I'])]
    q_h = hi['q'][(hi['I'] > 0) & np.isfinite(hi['I'])]
    I_h = hi['I'][(hi['I'] > 0) & np.isfinite(hi['I'])]

    q_ov_lo = max(q_l.min(), q_h.min())
    q_ov_hi = min(q_l.max(), q_h.max())

    if q_ov_lo >= q_ov_hi:
        mid   = (q_l.max() + q_h.min()) / 2.0
        scale = _scale_factor(q_l, I_l, q_h, I_h,
                               q_l.max() * 0.9, q_l.max() * 1.0)
        return float(mid), float(scale)

    q_grid = np.logspace(np.log10(q_ov_lo * 1.02),
                          np.log10(q_ov_hi * 0.98), 300)
    fl  = interp1d(q_l, I_l, bounds_error=False, fill_value=np.nan)
    fh  = interp1d(q_h, I_h, bounds_error=False, fill_value=np.nan)
    Il  = fl(q_grid)
    Ih  = fh(q_grid)
    ok  = np.isfinite(Il) & np.isfinite(Ih) & (Il > 0) & (Ih > 0)

    if ok.sum() < 10:
        scale = _scale_factor(q_l, I_l, q_h, I_h, q_ov_lo, q_ov_hi)
        return float((q_ov_lo + q_ov_hi) / 2), float(scale)

    q_ov  = q_grid[ok]
    Il_ov = Il[ok]
    Ih_ov = Ih[ok]

    # Restrict cut search to central 60 % of the overlap in log-q space.
    log_q   = np.log(q_ov)
    lq_lo   = log_q[0]  + 0.20 * (log_q[-1] - log_q[0])
    lq_hi   = log_q[-1] - 0.20 * (log_q[-1] - log_q[0])
    search  = (log_q >= lq_lo) & (log_q <= lq_hi)

    if search.sum() < 5:
        search = np.ones(len(q_ov), dtype=bool)   # fallback: use full overlap

    # Provisional scale from the high-q 40 % of the overlap.  The low-q edge
    # of the high-q detector often carries a beam-stop artefact (Ih >> Il),
    # which biases a full-overlap median and produces the wrong scale.  Using
    # only the upper part of the overlap avoids that contaminated region.
    hq_start  = log_q[0] + 0.60 * (log_q[-1] - log_q[0])
    hq_mask   = log_q >= hq_start
    if hq_mask.sum() >= 5:
        scale_prov = float(np.exp(np.median(np.log(Il_ov[hq_mask] / Ih_ov[hq_mask]))))
    else:
        scale_prov = float(np.exp(np.median(np.log(Il_ov / Ih_ov))))
    Ih_prov = Ih_ov * scale_prov

    dq  = float(np.mean(np.diff(log_q))) if len(log_q) > 1 else 1.0
    win = max(5, (search.sum() // 6) | 1)
    win = min(win, len(q_ov) - 2)
    win = win if win % 2 == 1 else win - 1
    win = max(win, 5)

    try:
        dIl = savgol_filter(np.log(Il_ov),   win, 2, deriv=1, delta=dq)
        dIh = savgol_filter(np.log(Ih_prov), win, 2, deriv=1, delta=dq)
    except Exception:
        dIl = np.gradient(np.log(Il_ov),   log_q)
        dIh = np.gradient(np.log(Ih_prov), log_q)

    score = (np.abs(np.log(Il_ov / Ih_prov))
             + 0.5 * np.abs(dIl - dIh) / (np.abs(dIl) + 1e-6))
    score_masked = np.where(search, score, np.inf)
    i_cut = int(np.argmin(score_masked))
    q_cut = float(q_ov[i_cut])

    # Final scale from a local window around q_cut — guarantees continuity at
    # the stitch point regardless of what the two branches do elsewhere.
    hw    = max(5, len(q_ov) // 10)
    sl    = slice(max(0, i_cut - hw), min(len(q_ov), i_cut + hw + 1))
    ratio = Il_ov[sl] / Ih_ov[sl]
    valid = ratio[np.isfinite(ratio) & (ratio > 0)]
    scale = float(np.exp(np.median(np.log(valid)))) if len(valid) >= 3 \
            else scale_prov

    return q_cut, scale


def _stitch_pair(q_l, I_l, sig_l, q_h, I_h, sig_h,
                 qcut: float | None, norm_exp: float = 1.0) -> tuple:
    """
    Stitch two (q, I, sig) arrays.  Returns (q, I, sig, qcut_used, scale).
    The high-q branch is scaled to match the low-q branch.
    """
    # Build temporary dicts for find_stitch_point
    lo = dict(q=q_l, I=I_l, sig=sig_l)
    hi = dict(q=q_h, I=I_h, sig=sig_h)

    if qcut is None:
        qcut, scale = find_stitch_point(lo, hi)
    else:
        vl = (I_l > 0) & np.isfinite(I_l)
        vh = (I_h > 0) & np.isfinite(I_h)
        q_ov_lo = max(q_l[vl].min(), q_h[vh].min())
        q_ov_hi = min(q_l[vl].max(), q_h[vh].max())
        scale   = _scale_factor(
            q_l[vl], I_l[vl], q_h[vh], I_h[vh],
            max(q_ov_lo, qcut * 0.7),
            min(q_ov_hi, qcut * 1.3))

    I_h_sc   = I_h   * scale
    sig_h_sc = sig_h * scale

    vl = (I_l > 0) & np.isfinite(I_l)
    vh = (I_h > 0) & np.isfinite(I_h)

    use_l = vl & (q_l <  qcut)
    use_h = vh & (q_h >= qcut)

    q_m   = np.concatenate([q_l[use_l],   q_h[use_h]])
    I_m   = np.concatenate([I_l[use_l],   I_h_sc[use_h]])
    sig_m = np.concatenate([sig_l[use_l], sig_h_sc[use_h]])
    return q_m, I_m, sig_m, qcut, scale


def stitch_chain(ordered_samples: list,
                 qcut: float | None = None,
                 normalize_exposure: bool = False) -> tuple:
    """
    Chain-stitch an ordered list of samples from lowest-q to highest-q.

    Returns (q_merged, I_merged, sig_merged, list_of_(qcut, scale) per join).
    """
    if len(ordered_samples) < 2:
        s = ordered_samples[0]
        v = _valid(s)
        return s['q'][v], s['I'][v], s['sig'][v], []

    def _arrays(s):
        exp = s['exposure'] if normalize_exposure else 1.0
        v   = _valid(s)
        return s['q'][v], s['I'][v] / exp, s['sig'][v] / exp

    q_acc, I_acc, sig_acc = _arrays(ordered_samples[0])
    joins = []

    for s in ordered_samples[1:]:
        q_h, I_h, sig_h = _arrays(s)
        q_acc, I_acc, sig_acc, qc, sc = _stitch_pair(
            q_acc, I_acc, sig_acc, q_h, I_h, sig_h, qcut)
        joins.append((qc, sc))

    return q_acc, I_acc, sig_acc, joins


# ── Stitched plots ─────────────────────────────────────────────────────────────

def _plot_stitched_overlay_core(results: list, ax_I, ax_snr):
    """Shared drawing logic for all stitched overlay variants."""
    colors = _colors(len(results))
    for item, c in zip(results, colors):
        q, I, sig, joins, lbl = item[0], item[1], item[2], item[3], item[4]
        v = (I > 0) & np.isfinite(I) & (sig > 0)
        ax_I.plot(q[v], I[v], color=c, lw=LW, label=lbl)
        if ax_snr is not None:
            ax_snr.plot(q[v], I[v]/sig[v], color=c, lw=LW)
        for qc, _ in joins:
            ax_I.axvline(qc, color=c, lw=0.6, ls=':', alpha=0.5)


def plot_stitched_overlay(results: list, outdir: Path):
    """results: list of (q, I, sig, joins, label, identity, temperature)."""
    if not results:
        return
    fig, ax_I, ax_snr = _make_Isnr_fig(8, 9)
    _plot_stitched_overlay_core(results, ax_I, ax_snr)
    _style_pair(ax_I, ax_snr, title='Stitched — all samples')
    ax_I.legend(fontsize=FS_LEG, frameon=True)
    fig.tight_layout(pad=0.2)
    out = outdir / 'stitched_overlay.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'    {out.relative_to(out.parents[4])}')


def plot_stitched_overlay_zoom(results: list, outdir: Path):
    """Like plot_stitched_overlay but zoomed to q > 1e-2 and I(q) < 1e-2, bottom 1e-5."""
    if not results:
        return
    fig, ax_I, ax_snr = _make_Isnr_fig(8, 9)
    _plot_stitched_overlay_core(results, ax_I, ax_snr)
    _style_pair(ax_I, ax_snr, title='Stitched — high-q zoom')
    ax_I.set_xlim(left=1e-2)
    ax_I.set_ylim(bottom=1e-5, top=1e-2)
    ax_I.legend(fontsize=FS_LEG, frameon=True)
    fig.tight_layout(pad=0.2)
    out = outdir / 'stitched_overlay_zoom.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'    {out.relative_to(out.parents[4])}')


def _save_grouped_overlay(relabelled: list, title: str, stem: str, outdir: Path,
                           xlim=None, ylim=None):
    """Draw one grouped overlay and save; optionally zoom with fixed axis limits."""
    fig, ax_I, ax_snr = _make_Isnr_fig(8, 9)
    _plot_stitched_overlay_core(relabelled, ax_I, ax_snr)
    _style_pair(ax_I, ax_snr, title=title)
    if xlim is not None:
        ax_I.set_xlim(*xlim)
        if ax_snr is not None:
            ax_snr.set_xlim(*xlim)
    if ylim is not None:
        ax_I.set_ylim(*ylim)
    ax_I.legend(fontsize=FS_LEG, frameon=True)
    fig.tight_layout(pad=0.2)
    out = outdir / f'{stem}.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'    {out.name}')


def plot_stitched_by_temperature(results: list, outdir: Path):
    """One overlay per temperature (full range + zoom) — curves labelled by sample identity."""
    by_temp: dict[str, list] = {}
    for item in results:
        by_temp.setdefault(item[6], []).append(item)
    if len(by_temp) < 2:
        return
    bt_dir      = ensure(outdir / 'by_temperature')
    bt_zoom_dir = ensure(outdir / 'by_temperature_zoom')
    for t, items in sorted(by_temp.items()):
        relabelled = [(q, I, s, j, ident, ident, tt)
                      for (q, I, s, j, _lbl, ident, tt) in items]
        _save_grouped_overlay(relabelled, f'Stitched — {t}', _safe(t), bt_dir)
        _save_grouped_overlay(relabelled, f'Stitched — {t} (zoom)',
                              _safe(t), bt_zoom_dir, xlim=(6e-2, 7e-1), ylim=(1e-5, 1e-2))


def plot_stitched_by_sample(results: list, outdir: Path):
    """One overlay per sample identity (full range + zoom) — curves labelled by temperature."""
    by_ident: dict[str, list] = {}
    for item in results:
        by_ident.setdefault(item[5], []).append(item)
    if len(by_ident) < 2:
        return
    bs_dir      = ensure(outdir / 'by_sample')
    bs_zoom_dir = ensure(outdir / 'by_sample_zoom')
    for ident, items in sorted(by_ident.items()):
        relabelled = [(q, I, s, j, tt, ident, tt)
                      for (q, I, s, j, _lbl, ident, tt) in items]
        _save_grouped_overlay(relabelled, f'Stitched — {ident}', _safe(ident), bs_dir)
        _save_grouped_overlay(relabelled, f'Stitched — {ident} (zoom)',
                              _safe(ident), bs_zoom_dir, xlim=(6e-2, 7e-1), ylim=(1e-5, 1e-2))


def plot_stitched_individual(q_m, I_m, sig_m, joins,
                              ordered_samples: list, range_order: list,
                              outdir: Path, title: str):
    fig, ax_I, ax_snr = _make_Isnr_fig(7, 8)

    # Original branches (faded) — one curve per range in this temperature group
    for i, s in enumerate(ordered_samples):
        rlabel = s['qrange']
        ls = _LS_CYCLE[i % len(_LS_CYCLE)]
        v  = _valid(s)
        ax_I.plot(s['q'][v], s['I'][v], color='gray',
                  lw=0.8, ls=ls, alpha=0.45,
                  label=f'orig {rlabel.replace("_","-")}')

    # Stitched
    v = (I_m > 0) & np.isfinite(I_m) & (sig_m > 0)
    ax_I.plot(q_m[v], I_m[v], color='#1f77b4', lw=LW, label='stitched')
    if ax_snr is not None:
        ax_snr.plot(q_m[v], I_m[v]/sig_m[v], color='#1f77b4', lw=LW)

    # Stitch points
    cut_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
    for k, (qc, sc) in enumerate(joins):
        cc = cut_colors[k % len(cut_colors)]
        ax_I.axvline(qc, color=cc, lw=0.8, ls='--',
                     label=f'$q_{{cut,{k+1}}}$ = {qc:.4f} Å⁻¹')
        if ax_snr is not None:
            ax_snr.axvline(qc, color=cc, lw=0.8, ls='--')

    _style_pair(ax_I, ax_snr, title=f'Stitched — {title}')
    ax_I.legend(fontsize=FS_LEG, frameon=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(outdir / f'{_safe(title)}.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ── Main per-folder pipeline ───────────────────────────────────────────────────

def _temp_key(s: dict) -> str:
    """Temperature label from a sample; strips any replicate suffix."""
    t = s['label'].rsplit(', ', 1)[-1]
    return re.sub(r'\s*\[rep \d+\]$', '', t).strip()


def process_folder(folder: Path,
                   qcut_manual: float | None = None,
                   normalize_exposure: bool = False,
                   blank_sub: bool = False,
                   blank_dir: Path | None = None):
    if not folder.exists():
        print(f'[WARN] Folder not found: {folder}', file=sys.stderr)
        return

    print(f'\n=== {folder} ===')

    dat_files = sorted(f for f in folder.glob('*.dat') if not _skip(f.name))
    if not dat_files:
        print('  No valid .dat files found.')
        return

    samples = []
    for f in dat_files:
        try:
            samples.append(load_saxs(f))
        except Exception as exc:
            print(f'  [WARN] {f.name}: {exc}')

    # Cluster into q-ranges
    range_groups = cluster_qranges(samples)
    range_order  = list(range_groups.keys())   # lowest-q first
    print(f'  Q-range groups ({len(range_order)}): '
          + ', '.join(f'{r} ({len(range_groups[r])} files)' for r in range_order))

    # Blank subtraction
    plots_suffix = ''
    if blank_sub or blank_dir is not None:
        if blank_dir is not None:
            if not blank_dir.exists():
                print(f'  [WARN] Blank folder not found: {blank_dir}; skipping subtraction.',
                      file=sys.stderr)
                blank_samples = []
            else:
                bl_files = sorted(f for f in blank_dir.glob('*.dat') if not _skip(f.name))
                bl_all = []
                for f in bl_files:
                    try:
                        bl_all.append(load_saxs(f))
                    except Exception as exc:
                        print(f'  [WARN] blank {f.name}: {exc}')
                # Prefer MQ water entries; fall back to all files
                blank_samples = [s for s in bl_all if s['sample']['type'] == 'mq'] or bl_all
                assign_qranges_from_ref(blank_samples, range_groups)
                print(f'  Blank folder: {blank_dir} '
                      f'({len(blank_samples)} blank sample(s))')
        else:
            blank_samples = [s for s in samples if s['sample']['type'] == 'mq']
            print(f'  Blank subtraction: {len(blank_samples)} MQ water sample(s) '
                  f'found in folder')

        if blank_samples:
            samples = subtract_blank(samples, blank_samples)
            range_groups = cluster_qranges(samples)
            range_order  = list(range_groups.keys())
            plots_suffix = '_nb'
        else:
            print('  [WARN] No blank samples found; skipping subtraction.')

    # Detect replicates within each q-range group
    rep_counter: dict[str, int] = {}
    for s in samples:
        key = f'{s["sample"]["position"]}|{s["sample"]["label"]}|{s["qrange"]}'
        rep_counter[key] = rep_counter.get(key, 0) + 1

    rep_seen: dict[str, int] = {}
    for s in samples:
        key = f'{s["sample"]["position"]}|{s["sample"]["label"]}|{s["qrange"]}'
        if rep_counter[key] > 1:
            rep_seen[key] = rep_seen.get(key, 0) + 1
            s['label'] += f' [rep {rep_seen[key]}]'

    n_rep_groups = sum(1 for v in rep_counter.values() if v > 1)
    if n_rep_groups:
        print(f'  Replicate groups: {n_rep_groups}')

    # Output directories
    plots   = make_plots_dir(folder, plots_suffix)
    ov_dir  = ensure(plots / 'overlay')
    ind_dir = plots / 'individuals'
    ool_dir = ensure(plots / 'overlap' / 'overlay')
    ooi_dir = ensure(plots / 'overlap' / 'individuals')
    stl_dir = ensure(plots / 'stitched' / 'overlay')
    sti_dir = ensure(plots / 'stitched' / 'individuals')
    print(f'  Output: {plots}')

    # 1. Overlay per q-range
    for rlabel, rsamples in range_groups.items():
        plot_overlay(rsamples, rlabel, ov_dir)

    # 2. Individual plots
    for s in samples:
        qr_dir = ensure(ind_dir / s['qrange'])
        plot_individual(s, qr_dir)

    # 3 & 4. Overlap + stitching (group by physical sample identity)
    series_list = group_by_identity(range_groups)

    # Only keep series that have data in at least 2 ranges (meaningful overlap)
    multi_range = [ser for ser in series_list
                   if sum(1 for v in ser['ranges'].values() if v) >= 2]
    single_range = [ser for ser in series_list
                    if sum(1 for v in ser['ranges'].values() if v) < 2]

    if single_range:
        print(f'  Single-range samples (no overlap): '
              f'{[s["identity"] for s in single_range]}')

    plot_overlap_overlay(multi_range, range_order, ool_dir)
    for ser in series_list:
        plot_overlap_individual(ser, range_order, ooi_dir)

    stitch_results = []
    for ser in multi_range:
        # Group measurements by temperature; stitch each (sample, temperature) pair.
        # For datasets with true replicates at one temperature this produces one
        # result per identity (same as before); for multi-temperature datasets it
        # produces one stitched curve per temperature per sample.
        temp_ranges: dict[str, dict[str, dict]] = {}   # temp → {rlabel: sample}
        for rlabel in range_order:
            for s in ser['ranges'].get(rlabel, []):
                t = _temp_key(s)
                temp_ranges.setdefault(t, {})
                if rlabel not in temp_ranges[t]:   # first rep at this temp+range
                    temp_ranges[t][rlabel] = s

        ident_lbl = _display_identity(ser)
        for t in sorted(temp_ranges):
            rl_map  = temp_ranges[t]
            ordered = [rl_map[rl] for rl in range_order if rl in rl_map]
            if len(ordered) < 2:
                continue
            full_lbl = f'{ident_lbl}, {t}'
            try:
                q_m, I_m, sig_m, joins = stitch_chain(
                    ordered, qcut=qcut_manual,
                    normalize_exposure=normalize_exposure)
                stitch_results.append(
                    (q_m, I_m, sig_m, joins, full_lbl, ident_lbl, t))
                plot_stitched_individual(q_m, I_m, sig_m, joins,
                                          ordered, range_order, sti_dir,
                                          title=full_lbl)
                join_str = '  '.join(
                    f'q_cut={qc:.4f} Å⁻¹ scale='
                    + (f'{sc:.3f}' if 0.01 <= sc <= 100 else f'{sc:.2e}')
                    for qc, sc in joins)
                print(f'    stitched: {full_lbl}  {join_str}')
                if any(sc < 0.001 or sc > 1000 for _, sc in joins):
                    print(f'      [NOTE] extreme scale factor — ranges likely '
                          f'have different absolute calibrations')
            except Exception as exc:
                print(f'  [WARN] stitch failed for {full_lbl}: {exc}')

    plot_stitched_overlay(stitch_results, stl_dir)
    plot_stitched_overlay_zoom(stitch_results, stl_dir)
    plot_stitched_by_temperature(stitch_results, stl_dir)
    plot_stitched_by_sample(stitch_results, stl_dir)
    print(f'  Done.')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Plot SAXS I(q) / Sig(q) from EDF .dat files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('--data', required=True,
                    help='Comma-separated folder names, e.g. 260813,260820')
    ap.add_argument('--base-dir', default='.',
                    help='Directory containing the data folders (default: cwd)')
    ap.add_argument('--qcut', type=float, default=None,
                    help='Manual stitch q-value [Å⁻¹]; auto-detect if omitted')
    ap.add_argument('--normalize-exposure', action='store_true',
                    help='Divide I by exposure time before stitching '
                         '(only if data were NOT already flux-normalised)')
    ap.add_argument('--snr', action='store_true',
                    help='Add I(q)/σ(q) panel below each plot (default: off)')
    ap.add_argument('--blank_sub', action='store_true',
                    help='Subtract MQ water blank found within each data folder; '
                         'output goes to plots_nb/ instead of plots/')
    ap.add_argument('--blank', default=None, metavar='DIR',
                    help='Name of folder (inside --base-dir) to use as blank '
                         'instead of in-folder MQ water; implies --blank_sub')
    args = ap.parse_args()

    global _PLOT_SNR
    _PLOT_SNR = args.snr

    base       = Path(args.base_dir)
    folders    = [f.strip() for f in args.data.split(',')]
    blank_dir  = base / args.blank if args.blank else None
    do_blank   = args.blank_sub or (blank_dir is not None)

    for name in folders:
        process_folder(base / name,
                       qcut_manual=args.qcut,
                       normalize_exposure=args.normalize_exposure,
                       blank_sub=do_blank,
                       blank_dir=blank_dir)


if __name__ == '__main__':
    main()

"""Threshold rules on top of sp_feat.features().  Thresholds live in thresholds.json (chosen on tune seeds 0-9
by tune_thr.py and then frozen); nothing here is fitted on the evaluation seeds.

classify() returns (label, confidence, why).  Labels:
  'noise-only'  nothing above the floor at any scale (or only noise-like wiggle)
  'hopper'      >= N channels visited, one at a time, dwell >= a few frames
  'ofdm'        flat top + CP autocorrelation peak that is NOT a comb
  'carrier'     one narrow spectral line, resolution-limited width
  'narrow' / 'medium' / 'wide'   by 99 % occupied bandwidth
  'unknown'     floor not observable (signal fills the usable band) or evidence is contradictory
confidence is in [0,1]: how far the decisive statistic is from its threshold (a heuristic, NOT a probability).
"""
from __future__ import annotations

import json
import os

from sp_stream import default_cfg

HERE = os.path.dirname(os.path.abspath(__file__))


def load_thr(path=None):
    with open(path or os.path.join(HERE, 'thresholds.json')) as f:
        return json.load(f)


def _margin(x, thr, scale=1.0):
    """squash (x-thr)/scale into 0..1"""
    d = (x - thr) / (scale * max(abs(thr), 1e-9))
    return float(max(0.0, min(1.0, 0.5 + 0.5 * d)))


def is_hopper(F, T):
    n_ch, hops, conc, dw = F.get('hop_n_ch'), F.get('hop_n_hops'), F.get('hop_conc'), F.get('hop_dwell_s')
    if None in (n_ch, hops, conc) or dw is None:
        return False, 0.0
    dt = default_cfg(F['rate'])['frame'] / F['rate']
    ok = (n_ch >= T['hop_n_ch'] and hops >= T['hop_n_hops'] and conc <= T['hop_conc_max']
          and dw >= T['hop_dwell_min_frames'] * dt and F.get('hop_dwell_kind') == 'period')
    return bool(ok), (min(1.0, 0.5 + hops / (8.0 * T['hop_n_hops'])) if ok else 0.0)


def is_ofdm(F, T):
    z, A, comb, lagB = F.get('cp_z'), F.get('cp_A'), F.get('cp_comb'), F.get('cp_lag_B')
    if None in (z, A, lagB):
        return False, 0.0
    comb_ok = (comb is None) or comb <= T['cp_comb_max']
    ok = (z >= T['cp_z'] and A >= T['cp_A'] and comb_ok and lagB >= T['cp_lagB_min']
          and (F.get('cp_npk') or 0) <= T['cp_npk_max'])
    return bool(ok), (_margin(A, T['cp_A'], 3.0) if ok else 0.0)


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def _conf_bw(F, T, obw):
    """Confidence of a bandwidth-class label: (i) how far the duty-diluted in-band SNR estimate is above the
    point where the 99 % tails vanish into the floor, (ii) how far the width is from the nearest class boundary
    (in decades; 0.3 decade = a factor 2 = full marks).  A heuristic score, not a probability."""
    import math
    c_snr = _clip01(((F.get('snr_est_db') if F.get('snr_est_db') is not None else -9.0) + 8.0) / 8.0)
    d = min(abs(math.log10(obw / T['bw_narrow_hz'])), abs(math.log10(obw / T['bw_wide_hz'])))
    return c_snr * _clip01(d / 0.3)


def classify(F, T):
    if not F.get('floor_ok'):
        # no visible noise floor (signal fills the usable band): only floor-free evidence can still speak.
        ok, c = is_ofdm(F, T)
        if ok:
            return 'ofdm', c, ['floor not observable; cp_z %.0f A %.3f lag %s' % (F['cp_z'], F['cp_A'], F['cp_lag'])]
        return 'unknown', 0.0, ['floor not observable']
    zs = F.get('struct_z')
    if zs is None or zs < T['z_struct']:
        return 'noise-only', _clip01((T['z_struct'] - (zs or 0.0)) / T['z_struct']), ['struct_z %.1f < %.1f' % (zs or 0, T['z_struct'])]
    ok, c = is_hopper(F, T)
    if ok:
        return 'hopper', c, ['hop channels %s hops %s' % (F.get('hop_n_ch'), F.get('hop_n_hops'))]
    ok, c = is_ofdm(F, T)
    if ok:
        return 'ofdm', c, ['cp_z %.0f A %.3f lag %s' % (F['cp_z'], F['cp_A'], F['cp_lag'])]
    obw = F.get('obw99_hz')
    if not obw:
        return 'unknown', 0.0, ['structure present but no measurable bandwidth']
    if F.get('n_lines') == 1 and (F.get('obw_bins') or 99) <= T['carrier_bins']:
        # a lone line is a carrier OR the only visible part of a modulated signal: confidence limited accordingly
        return 'carrier', 0.7, ['one line, width %.1f bins' % F['obw_bins']]
    c = _conf_bw(F, T, obw)
    if obw < T['bw_narrow_hz']:
        return 'narrow', c, ['obw99 %.0f Hz' % obw]
    if obw < T['bw_wide_hz']:
        return 'medium', c, ['obw99 %.0f Hz' % obw]
    return 'wide', c, ['obw99 %.0f Hz' % obw]

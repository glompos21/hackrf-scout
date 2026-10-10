"""Threshold selection on the TUNING seeds (0-9) only.

utility per capture:
    correct label of a signal           +1      (+1 more if a baud was answered within 5 % of the truth)
    refusing / 'none' on a signal        0
    wrong family label                  -4
    answered baud off by > 5 % (or any baud for a class without one)   -2
    noise or an 'other' class (lora, ofdm, hopper, pulsed, multi) refused  +1
    noise or 'other' labelled with a family  -4  (noise false alarm -6)
Coordinate descent over a small grid; each parameter is set to the CENTRE of the plateau of near-optimal
values (not the edge), then the final thresholds are written to thresholds.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np

import classify as C
import evaluate as E

GRID = dict(
    env_const=[0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03],
    env_var=[0.02, 0.03, 0.05, 0.08, 0.12],
    psk_env=[0.004, 0.008, 0.012, 0.016, 0.02],
    psk_z=[8, 15, 30, 60, 120, 250],
    psk_ratio=[2, 3, 5, 8, 12],
    fsk_z=[8, 15, 30, 60, 120, 250],
    fsk_ratio=[3, 5, 8, 12, 20, 30],
    carrier_peak3=[0.8, 0.85, 0.9, 0.95],
    am_peak3=[0.62, 0.66, 0.7, 0.74],
    am_peak3_hi=[0.8, 0.85, 0.9, 0.95],
    fsk_ratio_min=[3, 4, 5, 6],
    fsk_nh=[2, 3, 4],
    ook_sup=[0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8],
    ook_sup_label=[0.25, 0.3, 0.35, 0.4, 0.45],
    ook_sep=[0.64, 0.67, 0.7, 0.73, 0.76],
    ook_n1_max=[0.8, 0.85, 0.9, 0.95],
    pulse_top2=[0.15, 0.2, 0.25, 0.35],
    fm_snr_min=[-1, 1, 3, 5, 7],
    fm_bc_max=[0.35, 0.4, 0.45, 0.5, 0.55],
    fsk_ratio_solo=[10, 12, 15, 20, 30],
    fsk_z_solo=[50, 100, 200],
    chirp_max=[0.2, 0.3, 0.4, 0.5, 0.7],
    snr_min=[-6, -3, 0, 2],
    band2=[0.05, 0.15, 0.3, 0.5],
)


def utility_one(d, out):
    sc = E.score(d, out)
    fam = sc['truth']
    lab = out['label']
    u = 0.0
    if fam == 'none':
        u += 1.0 if lab in ('none', 'unknown') else -6.0
    elif fam in E.FAMS:
        if lab == fam:
            u += 1.0
        elif lab in ('none', 'unknown'):
            u += 0.0
        else:
            u -= 4.0
    else:
        if lab in ('none', 'unknown'):
            u += 1.0
        else:
            u -= 4.0
    if sc.get('has_rate_truth') and lab == fam:
        if sc['baud_answered']:
            u += 1.0 if sc['hit5'] else -2.0
    if sc['false_baud']:
        u -= 2.0
    return u


def total_utility(recs, thr):
    return sum(utility_one(d, C.classify(d['feat'], thr)) for d in recs)


def tune(recs, rounds=3, verbose=True):
    thr = dict(C.THR)
    best = total_utility(recs, thr)
    if verbose:
        print('start utility %.1f (n=%d)' % (best, len(recs)))
    for r in range(rounds):
        changed = False
        for k, grid in GRID.items():
            res = []
            for v in grid:
                th = dict(thr)
                th[k] = v
                res.append(total_utility(recs, th))
            m = max(res)
            # plateau: all grid values within 0.3 % of the best; take the middle one
            near = [i for i, x in enumerate(res) if x >= m - 0.003 * abs(m) - 0.5]
            pick = grid[near[len(near) // 2]]
            if pick != thr[k]:
                th = dict(thr)
                th[k] = pick
                if total_utility(recs, th) >= best - 1e-9:
                    thr = th
                    changed = True
            best = total_utility(recs, thr)
            if verbose:
                print('round %d %-14s -> %-6s  utility %.1f   (grid %s)' % (r, k, thr[k], best, ' '.join('%.0f' % x for x in res)))
        if not changed:
            break
    return thr, best


if __name__ == '__main__':
    path = sys.argv[1]
    recs = E.load(path)
    # tuning only uses the main variants
    recs = [d for d in recs if not d['variant'].startswith('x_')]
    print(len(recs), 'captures')
    thr, best = tune(recs)
    json.dump({k: thr[k] for k in GRID}, open(sys.argv[2], 'w'), indent=1)
    print('final utility %.1f -> %s' % (best, sys.argv[2]))

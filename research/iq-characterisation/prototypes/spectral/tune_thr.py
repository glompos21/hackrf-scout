"""Choose thresholds on the TUNE seeds (0-9) only and freeze them in thresholds.json.

Every threshold is printed with the evidence it was chosen from: the value range of the feature on the
classes it must separate.  Where a number is a convention (bandwidth class boundaries) it says so.
Run:  python3 tune_thr.py results/tune.jsonl
"""
from __future__ import annotations

import json
import sys

import numpy as np

import harness
import rules
import summ
import truth_class
from sp_stream import default_cfg


def kind_of(r):
    return harness.CASES[r['case']][0]


def main(path):
    rows = summ.load(path)
    sig = [r for r in rows if kind_of(r) != 'noise']
    noise = [r for r in rows if kind_of(r) == 'noise']
    T = {}

    # ---- 1. noise-only / structure test ----------------------------------------------------------
    zmax = max(r['feat']['struct_z'] for r in noise)
    T['z_struct'] = max(5.5, float(np.ceil((zmax + 0.5) * 2) / 2))
    print('noise captures %d: max struct_z %.2f -> z_struct %.1f' % (len(noise), zmax, T['z_struct']))

    # ---- 2. carrier ------------------------------------------------------------------------------
    cw = [r for r in sig if kind_of(r) == 'cw' and r['feat'].get('obw_bins')]
    ob = np.array([r['feat']['obw_bins'] for r in cw])
    other = np.array([r['feat']['obw_bins'] for r in sig if kind_of(r) not in ('cw',) and r['feat'].get('n_lines') == 1
                      and r['feat'].get('obw_bins')])
    print('carrier: cw obw_bins %.1f..%.1f (p90 %.1f); other single-line signals obw_bins min %.1f'
          % (ob.min(), ob.max(), np.percentile(ob, 90), other.min() if len(other) else np.nan))
    T['carrier_bins'] = 12.0   # cw p90 ~7 bins; the narrowest real modulated signals are >= 50 bins

    # ---- 3. bandwidth class boundaries = conventions (geometric midpoints of the observed clusters) --
    T['bw_narrow_hz'] = 25e3
    T['bw_wide_hz'] = 450e3

    # ---- 4. hopper ---------------------------------------------------------------------------------
    T.update(hop_n_ch=3, hop_n_hops=8, hop_conc_max=0.2, hop_dwell_min_frames=3)
    # ---- 5. OFDM ----------------------------------------------------------------------------------
    T.update(cp_z=10.0, cp_A=0.02, cp_comb_max=0.5, cp_lagB_min=24.0, cp_npk_max=6)

    # evidence for 4/5: positives per truth kind on the tune set (SNR >= 5 dB)
    def hop_rule(F):
        n_ch, hops, conc, dw = F.get('hop_n_ch'), F.get('hop_n_hops'), F.get('hop_conc'), F.get('hop_dwell_s')
        if None in (n_ch, hops, conc) or dw is None:
            return False
        dt = default_cfg(F['rate'])['frame'] / F['rate']
        return (n_ch >= T['hop_n_ch'] and hops >= T['hop_n_hops'] and conc <= T['hop_conc_max']
                and dw >= T['hop_dwell_min_frames'] * dt and F.get('hop_dwell_kind') == 'period')

    def ofdm_rule(F):
        z, A, comb, lagB = F.get('cp_z'), F.get('cp_A'), F.get('cp_comb'), F.get('cp_lag_B')
        if None in (z, A, lagB):
            return False
        return (z >= T['cp_z'] and A >= T['cp_A'] and (comb is None or comb <= T['cp_comb_max'])
                and lagB >= T['cp_lagB_min'] and (F.get('cp_npk') or 0) <= T['cp_npk_max'])

    for name, fn in (('hopper rule', hop_rule), ('ofdm rule', ofdm_rule)):
        print('\n%s: positives by case (clean / imp), all SNR' % name)
        for case in harness.CASES:
            for cond in ('clean', 'imp'):
                rr = [r for r in rows if r['case'] == case and r['cond'] == cond]
                if rr:
                    k = sum(fn(r['feat']) for r in rr)
                    if k:
                        print('  %-15s %-5s %3d / %3d' % (case, cond, k, len(rr)))
    T['_provenance'] = ('tune seeds 0-9 only; ' + path + '; conventions: bw_narrow_hz, bw_wide_hz; '
                        'z_struct = max(5.5, noise max + 0.5)')
    with open('thresholds.json', 'w') as f:
        json.dump(T, f, indent=1)
    print('\nwritten thresholds.json', {k: v for k, v in T.items() if not k.startswith('_')})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/tune.jsonl')

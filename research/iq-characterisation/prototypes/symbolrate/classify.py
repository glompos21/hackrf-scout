"""Rule-based family label + symbol-rate hint from the feature dict (features.analyze).

classify(feat) -> dict(label, baud_hz, bound_rel, kind, conf, reasons, ...)

label in {'none', 'unknown', 'carrier', 'am', 'fm', 'ook', 'fsk', 'psk'}
  none     no signal found above the noise floor (also what pure noise must produce)
  unknown  signal present but the evidence does not support any family (refusal)
All thresholds live in THR so the tuning script can rewrite them (thresholds.json).
"""
from __future__ import annotations

import json
import math
import os

THR = dict(
    band2=0.30,          # second band's excess power / first band's excess power -> multi-signal -> unknown
    env_const=0.015,     # env_exc below this -> constant envelope
    env_var=0.05,        # env_exc above this -> varying envelope (zone in between -> undecided)
    psk_env=0.012,       # PSK/QAM needs at least this envelope excess (a constant-envelope signal with a line is FSK-like)
    pulse_env=0.8,       # env_exc above this with a tiny duty -> pulse train
    pulse_top2=0.25,     # share of the energy in the strongest 2 % of samples (Gaussian noise: 0.09; OOK: 0.05-0.08)
    pulse_duty=0.08,
    psk_z=30.0,          # |y|^2 cyclic line z-score
    psk_ratio=5.0,
    psk_rel_lo=0.45, psk_rel_hi=1.05,    # f0 / band_bw
    fsk_z=30.0,          # delay-multiply fundamental z-score
    fsk_ratio=8.0,       # ... and line-to-local-floor ratio (z alone is fooled by long captures: WFM pilot ratio 3-5)
    fsk_ratio_min=4.0,   # ... unless the line has >= fsk_nh harmonics (a data-clock comb), then this lower ratio is enough
    fsk_nh=3,
    fsk_rel_lo=0.015, fsk_rel_hi=1.05,
    conflict_env_max=0.10,  # the |y|^2-vs-delay-multiply conflict test only applies to weakly envelope-modulated signals (OOK/AM lines are unrelated)
    cross_tol=0.02,      # |y|^2 line and delay-multiply line must agree within this fraction
    dm_agree_tol=0.01,   # both delays must give the same fundamental within this fraction
    fsk_ratio_solo=12.0,  # ... unless a single-delay line is this strong (high-rate GFSK: the longer delay loses the line)
    fsk_z_solo=100.0,
    chirp_max=0.35,      # |P(slope>0)-0.5|*2 of the smoothed discriminator above this: linear sweep (LoRa/FMCW), not data
    periodic_acf=0.70,   # |acf of discriminator at lag T| above this -> deterministic (tone / chirp)
    carrier_peak3=0.90,
    carrier_bw_max=30e3,  # physical plausibility: an unmodulated carrier / narrow AM / voice FM / FM broadcast cannot be wider than this
    am_bw_max=60e3,
    fm_bw_max=400e3,
    fm_bc_max=0.45,      # analog FM: instantaneous-frequency histogram is unimodal; flat (chirp) or two-humped (data) is not
    am_peak3=0.70,       # carrier share needed for AM when the envelope is periodic (tone)
    am_peak3_hi=0.85,    # carrier share that is enough on its own (voice / music AM)
    ook_sup=0.60,        # edge-interval support needed to quote a chip rate
    ook_sup_label=0.40,
    ook_sep=0.70,        # Otsu between-class variance share of the smoothed envelope (Rayleigh/noise-like: 0.64, two-level: 0.75+)
    ook_n1_max=0.90,     # fraction of unit-length runs above this -> periodic (tone-like) envelope
    ook_duty_lo=0.08, ook_duty_hi=0.92,
    fm_snr_min=3.0,      # 'analog FM' = constant envelope without a symbol line; only meaningful if a line would have shown
    snr_min=-3.0,        # psd-based SNR floor below which nothing is claimed
)

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thresholds.json')
if os.path.exists(_path):
    THR.update(json.load(open(_path)))

# standard / common rates a plausibility check can compare with (symbols or chips per second)
STD_RATES = [110, 134.5, 300, 512, 600, 1200, 1600, 1800, 2400, 3200, 4800, 6400, 9600, 14400, 19200, 28800, 38400,
             57600, 76800, 115200, 125000, 230400, 250000, 460800, 500000, 921600, 1000000, 2000000]


def plausible_rates(baud, tol):
    return [r for r in STD_RATES if abs(r - baud) <= tol * baud]


def _best(lst, key):
    return max(lst, key=lambda x: x[key]) if lst else None


def _fund_bound(L, k=0.25, clock=1e-5):
    """Relative half-width of a line-derived rate: a quarter of a (zero-padded) FFT bin plus 10 ppm clock slop.
    (Tuning seeds: the real error was <= 0.02 bin.)"""
    return k * L['df'] / L['f0'] + clock


def ook_bound(ook):
    """Edge-interval estimate: 0.4 % when the intervals sit on a clean grid, growing to ~5 % for support 0.55."""
    return 0.004 + 0.05 * max(0.0, 0.9 - ook['support']) / 0.35


def classify(f, thr=None):
    t = dict(THR if thr is None else thr)
    out = dict(label='unknown', baud_hz=None, bound_rel=None, kind=None, reasons=[], ev={}, plausible=[],
               hints={})
    why = out['reasons']
    st = f.get('stage')
    if st == 'nosignal':
        out['label'] = 'none'
        why.append('no band above the noise floor')
        return out
    if st == 'wideband':
        why.append('signal wider than 55%% of the capture band (%.2f)' % f.get('rel_bw', 0))
        return out
    if f.get('band2_rel', 0) >= t['band2']:
        why.append('several comparable bands (second/first = %.2f): multi-signal, hopping or multi-carrier' % f['band2_rel'])
        return out
    snr0 = f['snr_psd_db'] - 10.0 * math.log10(max(min(f.get('blk_duty', 1.0), 1.0), 0.02))
    if snr0 < t['snr_min']:
        why.append('in-band SNR %.1f dB below the analysis floor' % snr0)
        return out

    bw = f['band_bw']
    env = f['env_exc']
    ef = f.get('env_fund')
    dmf = _best(f.get('dm_fund') or [], 'z')
    # A strong carrier in noise makes the delay-multiply spectrum ripple with period fs/tau: that comb looks like
    # a line family.  A real symbol clock gives the SAME fundamental for both delays (tau and 2 tau).
    dm_agree = bool(dmf and any(x is not dmf and abs(x['f0'] - dmf['f0']) <= t['dm_agree_tol'] * dmf['f0']
                                for x in (f.get('dm_fund') or [])))
    ook = f.get('ook') or {}
    peak3 = f['peak3_frac']
    acfT = f.get('disc_acf_T')
    # in-band SNR of the ON state: the capture-averaged PSD excess is diluted by the share of active blocks
    snr = f['snr_psd_db'] - 10.0 * math.log10(max(min(f.get('blk_duty', 1.0), 1.0), 0.02))
    ev = out['ev']
    ev.update(env_exc=env, peak3=peak3, snr_psd_db=snr)

    # ---- pulse train (radar-like): peaky envelope with a tiny duty
    if env >= t['pulse_env'] and f.get('top2_frac', 0) >= t['pulse_top2'] and ook.get('duty_otsu', 1.0) < t['pulse_duty']:
        why.append('pulse train: envelope excess %.2f, top-2%% energy share %.2f, duty %.3f'
                   % (env, f.get('top2_frac', 0), ook.get('duty_otsu', 0)))
        return out

    # cross-estimator consistency: an envelope (|y|^2) cyclic line and a delay-multiply line that are both strong
    # but at different rates mean we do not know the clock
    if (ef and dmf and env < t['conflict_env_max'] and ef['z'] >= t['psk_z'] and dmf['z'] >= t['fsk_z']
            and dmf['ratio'] >= t['fsk_ratio_solo']):
        def _related(a, b):          # equal, or one is an integer multiple (<= 40) of the other
            r = a / b if a >= b else b / a
            return abs(r - round(r)) <= t['cross_tol'] * r and round(r) <= 40
        agree_any = _related(dmf['f0'], ef['f0'])
        if not agree_any:
            why.append('conflicting rate evidence: |y|^2 line at %.1f Hz vs delay-multiply line at %.1f Hz' % (ef['f0'], dmf['f0']))
            return out

    # ---- linear modulation (PSK / QAM): strong cyclic line at ~ bandwidth/(1+beta) in |y|^2
    if (ef and ef['z'] >= t['psk_z'] and ef['ratio'] >= t['psk_ratio'] and t['psk_rel_lo'] <= ef['f0'] / bw <= t['psk_rel_hi']
            and env > t['psk_env']):
        out.update(label='psk', baud_hz=ef['f0'], kind='symbol', bound_rel=_fund_bound(ef))
        ev.update(line_z=ef['z'], line_ratio=ef['ratio'], f0_over_bw=ef['f0'] / bw)
        x2, x4 = f.get('x2_line'), f.get('x4_line')
        if x2 and x4:
            if x2['ratio'] >= 20 and x2['delta'] >= 10:
                out['hints']['order'] = 'BPSK-like (squaring line)'
            elif x4['ratio'] >= 20 and x4['delta'] >= 10:
                out['hints']['order'] = 'QPSK-like (4th-power line)'
            else:
                out['hints']['order'] = 'no squaring/4th-power line: higher order or filtered'
        why.append('cyclic line in |y|^2 at %.1f Hz (z=%.0f, ratio %.0f), %.2f of the bandwidth; envelope excess %.3f'
                   % (ef['f0'], ef['z'], ef['ratio'], ef['f0'] / bw, env))
        _finish(out, f)
        return out

    # ---- constant envelope branch: carrier / FSK / analog FM / chirp
    if env <= t['env_const']:
        if f.get('chirp_bias', 0.0) >= t['chirp_max']:
            why.append('constant envelope with a one-signed frequency slope (chirp bias %.2f): linear sweep, not data FSK or analog FM'
                       % f['chirp_bias'])
            return out
        line_ok = bool(dmf and (dm_agree or (dmf['ratio'] >= t['fsk_ratio_solo'] and dmf['z'] >= t['fsk_z_solo'])) and dmf['z'] >= t['fsk_z'] and (dmf['ratio'] >= t['fsk_ratio'] or
                                                           (dmf['ratio'] >= t['fsk_ratio_min'] and dmf['n_harm'] >= t['fsk_nh'])))
        if line_ok and t['fsk_rel_lo'] <= dmf['f0'] / bw <= t['fsk_rel_hi']:
            if acfT is not None and abs(acfT) >= t['periodic_acf']:
                why.append('constant envelope, line at %.1f Hz but the discriminator repeats at that lag (acf %.2f): tone/chirp, not data'
                           % (dmf['f0'], acfT))
                return out
            out.update(label='fsk', baud_hz=dmf['f0'], kind='symbol', bound_rel=_fund_bound(dmf))
            ev.update(line_z=dmf['z'], line_ratio=dmf['ratio'], n_harm=dmf['n_harm'], f0_over_bw=dmf['f0'] / bw)
            if 'disc_q' in f:
                dev = 0.5 * (f['disc_q'][1] + f['disc_q'][2])       # mean of the median and 90th percentile of |f - f_centre|
                out['hints']['dev_hz'] = dev
                out['hints']['h'] = 2.0 * dev / dmf['f0']
            why.append('constant envelope, delay-multiply line at %.1f Hz (z=%.0f, %d harmonics), acf(T)=%s'
                       % (dmf['f0'], dmf['z'], dmf['n_harm'], 'n/a' if acfT is None else '%.2f' % acfT))
            _finish(out, f)
            return out
        if peak3 >= t['carrier_peak3'] and bw <= t['carrier_bw_max']:
            out['label'] = 'carrier'
            why.append('constant envelope, %.0f%% of the power in 3 bins' % (100 * peak3))
            return out
        if (not line_ok and snr >= t['fm_snr_min'] and bw <= t['fm_bw_max'] and f.get('disc_bc', 0.0) <= t['fm_bc_max']):
            out['label'] = 'fm'
            why.append('constant envelope, no symbol line, power spread over %.0f kHz' % (bw / 1e3))
            return out
        if not line_ok:
            why.append('constant envelope, no data line; not claimed as analog FM (SNR %.1f dB, width %.0f kHz, freq-histogram bc %.2f)'
                       % (snr, bw / 1e3, f.get('disc_bc', float('nan'))))
        else:
            why.append('constant envelope with a line at an implausible fraction of the bandwidth')
        return out

    # ---- varying envelope branch: OOK / AM
    if env >= t['env_var']:
        sup = ook.get('support')
        two_level = bool('T' in ook and sup >= t['ook_sup_label'] and ook['sep'] >= t['ook_sep']
                         and t['ook_duty_lo'] <= ook.get('duty_hi', 0.5) <= t['ook_duty_hi'])
        clean_grid = bool(two_level and sup >= t['ook_sup'] and ook['frac_n1'] < t['ook_n1_max'])
        periodic_env = bool(sup is not None and sup >= t['ook_sup'] and ook.get('frac_n1', 0) >= t['ook_n1_max'])
        if clean_grid:
            # edges on an integer grid of one unit interval, aperiodic: keyed carrier (checked BEFORE the carrier test
            # because a slow OOK signal is narrower than the PSD resolution and looks like 'a carrier' there)
            out.update(label='ook', kind='chip', baud_hz=1.0 / ook['T'], bound_rel=ook_bound(ook))
            ev.update(support=sup, rms=ook['rms'], n_used=ook['n_used'], frac_n1=ook['frac_n1'])
            ef2 = f.get('env_fund')
            if ef2 and ef2['z'] >= 15:
                k = (1.0 / ook['T']) / ef2['f0']
                if abs(k - round(k)) < 0.06 * max(k, 1) and 1 <= round(k) <= 12:
                    out['hints']['chips_per_period'] = int(round(k))
                    out['hints']['period_hz'] = ef2['f0']
            if ook.get('run_hist'):
                h = ook['run_hist']
                out['hints']['run_hist_1T_2T_3T_4T+'] = [round(x, 3) for x in h]
                out['hints']['coding_guess'] = ('PWM-like (1T and 2T runs, none longer)' if h[1] >= 0.33 and h[2] < 0.03
                                                else 'Manchester-like (<=2T runs)' if h[2] + h[3] < 0.04
                                                else 'NRZ-like (runs of 3T and longer)')
            why.append('envelope edges at integer multiples of %.1f us (support %.2f, %d runs)' %
                       (ook['T'] * 1e6, sup, ook['n_used']))
            _finish(out, f)
            return out
        if (peak3 >= t['am_peak3_hi'] or (peak3 >= t['am_peak3'] and periodic_env)) and bw <= t['am_bw_max']:
            out['label'] = 'am'
            why.append('envelope excess %.2f with a carrier line (%.0f%% of power in 3 bins)%s' %
                       (env, 100 * peak3, ', periodic envelope' if periodic_env else ''))
            return out
        if two_level:
            out.update(label='ook', kind='chip')
            why.append('two-level envelope but edge intervals not on a clean grid (support %.2f): no baud quoted' % sup)
            ef2 = f.get('env_fund')
            if ef2 and ef2['z'] >= 15:
                out['hints']['period_hz'] = ef2['f0']
            return out
        why.append('varying envelope (%.2f) but neither a clean two-level pattern nor a carrier' % env)
        return out

    why.append('envelope excess %.3f in the undecided zone' % env)
    return out


def _finish(out, f):
    """Plausibility against standard rates (never changes the estimate)."""
    b, bound = out['baud_hz'], out['bound_rel']
    if b is None:
        return
    out['plausible'] = plausible_rates(b, max(bound, 0.0))

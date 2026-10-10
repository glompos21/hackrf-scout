"""Rule-based labeller on top of the amplitude features (works on the flattened bench rows, i.e. on features only).

Every label carries a confidence in [0, 1] and `UNKNOWN` is a first-class answer.  Thresholds are physics-derived
midpoints (constant envelope m4 = 1, Gaussian-like m4 = 2, noise-compensated) and were only *checked* on the tuning
seeds 0-9.

Labels
    NOISE          nothing above the floor in the PSD, no spectral line, no energy burst
    CARRIER        continuous, constant envelope, one spectral line holds >= 85 % of the in-band excess
    CE_CONT        continuous constant envelope without a line (FM, FSK, ...)
    CE_BURSTY      on-off structure with a constant-envelope on-state (OOK, packetised FSK/LoRa, gated carriers)
    LIN_CONT/LIN_BURSTY    envelope intermediate between constant and Gaussian (AM, RRC-shaped PSK)
    GAUSS_CONT/GAUSS_BURSTY  Gaussian-like envelope (OFDM, noise-like)
    PULSED         short (<= 100 us) bursts at <= 6 % duty
    UNKNOWN        evidence insufficient (low channel SNR, big standard error, conflicting tests)
"""
import math

TH = dict(
    m4_ce=1.10,            # CE is 1.00, RRC-QPSK 1.20, RRC-BPSK 1.37: midpoint of CE and the nearest linear modulation
    m4_gauss=1.88,         # between AM tone (~1.76) and OFDM (~2.0)
    snr_ch_min_db=2.0,     # on-state SNR in the channel below which the envelope shape is not trusted
    se_max=0.20,           # jackknife se of m4_sig above which the shape is not trusted
    duty_cont=0.85,
    pulsed_duty=0.06,
    pulsed_run_s=100e-6,
    carrier_frac=0.70,     # share of in-channel excess power in a 5-bin (~25 Hz) line: CW 0.86, AM 0.76-0.97, NFM voice 0.5, FSK/OOK <= 0.41
    carrier_z=10.0,
    clip_max=1e-3,         # raw clip fraction above which the envelope shape is not trusted (clipping flattens peaks)
    seg_ev_lams=12.0,      # median on-run evidence must exceed this many HMM switch costs (tune seeds: all wrong below ~8, mostly right above 16)
    llr_min=20.0,          # nats of burst evidence (HMM path vs all-off) needed to accept bursty structure
)


def _f(row, key, default=float('nan')):
    v = row.get(key, default)
    return default if v is None else v


def _isnan(x):
    return x != x


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def decide(row, th=None, use_oracle=False):
    th = th or TH
    out = dict(label='UNKNOWN', conf=0.0, shape='?', temporal='?', why='')
    if not row.get('has_channel', False):
        out.update(label='NOISE', conf=0.5, why='no psd cluster, no prior channel')
        return out
    in_win = abs(_f(row, 'line_f', 1e12) - _f(row, 'prior_fc', 0.0)) <= _f(row, 'prior_bw', 1e12)
    p_psd = row.get('ch_src') in ('psd', 'line', 'psd+prior')
    p_line = _f(row, 'line_over_thr_db', -9) > 0 and in_win
    p_time = bool(row.get('det_any', 0.0))
    p_line = p_line or (_f(row, 'car_carrier_over_thr_db', -9) > 0)
    if not (p_psd or p_line or p_time):
        out.update(label='NOISE', conf=0.9, why='no cluster / line / burst above floor')
        return out
    pre = 'o_' if use_oracle else ''
    duty = _f(row, 'o_duty' if use_oracle else 'duty')
    n_runs = _f(row, 'run_n_runs')
    run_med = _f(row, 'run_run_med_s')
    llr = _f(row, 'det_llr')
    # --- temporal -------------------------------------------------------------------------------
    if p_time and duty < th['duty_cont'] and (llr >= th['llr_min'] or use_oracle):
        out['temporal'] = 'BURSTY'
        if duty <= th['pulsed_duty'] and run_med <= th['pulsed_run_s'] and n_runs >= 5:
            out['temporal'] = 'PULSED'
    elif p_time or p_psd or p_line:
        out['temporal'] = 'CONT'
    # --- on-state shape -------------------------------------------------------------------------
    m4s = _f(row, pre + 'on_m4_sig')
    snr = _f(row, pre + 'on_snr_ch_db')
    se = _f(row, pre + 'onse_se_m4_sig')
    if out['temporal'] == 'CONT' and _isnan(m4s):
        m4s, snr, se = _f(row, pre + 'all_m4_sig'), _f(row, pre + 'all_snr_ch_db'), _f(row, 'onse_se_m4_sig')
    shape, conf = '?', 0.0
    clipped = _f(row, 'raw_clip_frac', 0.0) > th['clip_max']
    if clipped:
        out['why'] = 'clipping %.2f %%' % (100 * row['raw_clip_frac'])
    if (not clipped) and not _isnan(m4s) and not _isnan(snr) and snr >= th['snr_ch_min_db']:
        se_eff = se if not _isnan(se) else 0.1
        se_eff = max(se_eff, 0.02)
        if se_eff <= th['se_max']:
            if m4s < th['m4_ce']:
                shape, conf = 'CE', phi((th['m4_ce'] - m4s) / se_eff)
            elif m4s < th['m4_gauss']:
                shape, conf = 'LIN', phi(min(m4s - th['m4_ce'], th['m4_gauss'] - m4s) / se_eff)
            else:
                shape, conf = 'GAUSS', phi((m4s - th['m4_gauss']) / se_eff)
    out['shape'] = shape
    t = out['temporal']
    if t == 'BURSTY' and not use_oracle:
        # can the burst structure be trusted?  evidence of the median on-run (nats) vs 4x the HMM switch cost
        Lr = _f(row, 'det_L_over_N', 1.0)
        n_dof = run_med / max(_f(row, 'tick_s', 1.0), 1e-12) * _f(row, 'k_tick', 1.0)
        ev = n_dof * (Lr - 1.0 - math.log(max(Lr, 1.0001)))
        lam = 0.5 * math.log(max(_f(row, 'ch_T', 1000.0), 2.0)) + 4.0
        out['seg_evidence'] = ev
        if ev < th['seg_ev_lams'] * lam:
            out.update(label='UNKNOWN', conf=0.0, why='burst structure unresolved: median burst evidence %.0f nats' % ev)
            return out
    if t == 'PULSED':
        out.update(label='PULSED', conf=0.7, why='short bursts at low duty')
        return out
    if shape == '?':
        out.update(label='UNKNOWN', conf=0.0, why='shape not trusted (snr_ch %.1f dB, se %.2f)' % (snr, se if not _isnan(se) else -1))
        return out
    if t == 'CONT':
        if shape == 'CE':
            if _f(row, 'car_carrier_frac', 0.0) >= th['carrier_frac'] and _f(row, 'car_carrier_z', 0.0) >= th['carrier_z']:
                out.update(label='CARRIER', conf=conf)
            else:
                out.update(label='CE_CONT', conf=conf)
        else:
            out.update(label=shape + '_CONT', conf=conf)
    elif t == 'BURSTY':
        out.update(label=shape + '_BURSTY', conf=conf)
    return out


# ---- ground truth mapping (from the generator's own truth, not from any feature) -------------------------------
SHAPE_TRUE = {
    'cw': 'CE', 'nfm_voice': 'CE', 'nfm_tone': 'CE', 'wfm': 'CE', 'ook_pwm': 'CE', 'ook_man': 'CE', 'fsk2_rect': 'CE',
    'fsk2_gauss': 'CE', 'gfsk': 'CE', 'lora7': 'CE', 'lora9': 'CE', 'hopper': 'CE', 'pulsed': 'CE', 'pulsed_fast': 'CE',
    'pulsed_lfm': 'CE', 'am_tone': 'LIN', 'am_voice': 'LIN', 'bpsk': 'LIN', 'qpsk': 'LIN', 'ofdm_wide': 'GAUSS',
    'ofdm_narrow': 'GAUSS',
}


def true_label(row, duty_cont=0.9):
    s = row['scen']
    if s.startswith('noise'):
        return 'NOISE'
    if s.startswith('multi'):
        return None
    if s.startswith('pulsed'):
        return 'PULSED'
    shape = SHAPE_TRUE[s]
    if row['t_duty'] >= duty_cont:
        if s == 'cw':
            return 'CARRIER'
        return ('CE_CONT' if shape == 'CE' else shape + '_CONT')
    return shape + '_BURSTY'

"""Flatten the nested feature dict to a numeric vector (for inspection / CART / the rule classifier)."""
import math
import numpy as np

def _lg(x, floor=1e-3):
    return math.log10(max(x, floor)) if x is not None else float('nan')

def nz(x):
    return float('nan') if x is None else float(x)

def flatten(f):
    o = {}
    st = f.get('stage')
    o['stage_ok'] = 1.0 if st == 'ok' else 0.0
    if st != 'ok':
        return o
    bw = f['band_bw']
    o['snr_psd'] = f['snr_psd_db']; o['snr_on'] = f['snr_on_db']
    o['lbw'] = math.log10(bw); o['rel_bw'] = f['rel_bw']; o['bw99_bw'] = f['bw99'] / bw; o['bw90_bw'] = f['bw90'] / bw
    o['act'] = f['act']; o['weak'] = float(f['weak']); o['n_frames'] = f['n_frames']; o['band2'] = f['band2_rel']
    o['peak3'] = f['peak3_frac']; o['sigf_bw'] = f['sigma_f'] / bw; o['pkoff_bw'] = abs(f['pk_off']) / bw
    o['env_exc'] = f['env_exc']; o['env_V'] = f['env_V']
    # envelope lines
    ef = f.get('env_fund')
    if ef:
        o['envf_rel'] = ef['f0'] / f['bw90']; o['envf_lr'] = _lg(ef['ratio']); o['envf_z'] = _lg(ef['z']); o['envf_nh'] = ef['n_harm']
    el = f.get('env_lines') or []
    if el:
        o['envt_rel'] = el[0]['f'] / f['bw90']; o['envt_lr'] = _lg(el[0]['ratio'])
    for k in ('disc_sd', 'disc_mabs'):
        if k in f: o[k + '_bw'] = f[k] / bw
    for k in ('disc_bc', 'disc_ku', 'disc_centre_frac', 'disc_acf_T', 'disc_acf_T2', 'dm_sps'):
        if k in f: o[k] = f[k]
    if 'disc_acf_T' in f: o['abs_acfT'] = abs(f['disc_acf_T'])
    dmf = f.get('dm_fund') or []
    if dmf:
        b = max(dmf, key=lambda x: x['z'])
        o['dmf_rel'] = b['f0'] / f['bw90']; o['dmf_lr'] = _lg(b['ratio']); o['dmf_z'] = _lg(b['z']); o['dmf_nh'] = b['n_harm']
    dm = f.get('dm_lines') or []
    if dm:
        b = max(dm, key=lambda x: x['ratio'])
        o['dmt_rel'] = b['f'] / f['bw90']; o['dmt_lr'] = _lg(b['ratio'])
    for q in (2, 4):
        x = f.get('x%d_line' % q)
        if x:
            o['x%d_delta' % q] = _lg(x['delta']); o['x%d_lr' % q] = _lg(x['ratio'])
    ok = f.get('ook')
    if ok:
        o['ook_sep'] = ok['sep']; o['ook_floor'] = _lg(ok['floor_ratio']); o['ook_duty'] = ok.get('duty_hi', float('nan'))
        o['ook_lo_frac'] = ok['lo_frac']
        if 'T' in ok:
            o['ook_sup'] = ok['support']; o['ook_rms'] = ok['rms']; o['ook_n1'] = ok['frac_n1']; o['ook_rel'] = (1.0 / ok['T']) / f['bw90']
            o['ook_nused'] = ok['n_used']
    fr = f.get('fsk_runs')
    if fr and 'T' in fr:
        o['fskr_sup'] = fr['support']; o['fskr_n1'] = fr['frac_n1']
    return o

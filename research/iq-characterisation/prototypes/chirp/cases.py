"""Benchmark case definitions and the sweep-metadata model for the chirp / pulsed prototype.

Every case is drawn from its seed: offset error of the capture centre (the sweep centre is only good to
~ +-50 kHz), CFO of a few kHz, DC offset of a few LSB (I and Q different), on-fraction 10-30 %.
"""
from __future__ import annotations

import os
import zlib
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gen  # noqa: E402

# name -> dict(kind, kw, rate, pos_chirp, pos_pulse, group)
SPECS = {
    # ---- chirp spread spectrum (positives for the chirp flag) --------------------------------------
    'lora_sf7_125': dict(kind='lora', kw=dict(sf=7), rate=2e6, chirp=True, group='lora'),
    'lora_sf9_125': dict(kind='lora', kw=dict(sf=9), rate=2e6, chirp=True, group='lora'),
    'lora_sf12_125': dict(kind='lora', kw=dict(sf=12), rate=2e6, chirp=True, group='lora'),
    'lora_sf8_250': dict(kind='lora', kw=dict(sf=8, bw_hz=250e3), rate=2e6, chirp=True, group='lora'),
    'lora_sf7_500': dict(kind='lora', kw=dict(sf=7, bw_hz=500e3), rate=2e6, chirp=True, group='lora'),
    'lora_sf10_125': dict(kind='lora', kw=dict(sf=10), rate=2e6, chirp=True, group='lora'),
    'lora_sf6_125': dict(kind='lora', kw=dict(sf=6), rate=2e6, chirp=True, group='lora'),
    'lora_sf8_125_10M': dict(kind='lora', kw=dict(sf=8), rate=10e6, chirp=True, group='lora', secs=5.0),
    # ---- pulsed (positives for the pulsed flag) ----------------------------------------------------
    'pulsed_10us_1ms': dict(kind='pulsed', kw={}, rate=2e6, pulse=True, group='pulsed'),
    'pulsed_5us_2ms': dict(kind='pulsed', kw=dict(pw_s=5e-6, pri_s=2e-3), rate=2e6, pulse=True, group='pulsed'),
    'pulsed_20us_jit': dict(kind='pulsed', kw=dict(pw_s=20e-6, pri_s=800e-6, pri_jitter=0.05), rate=2e6, pulse=True, group='pulsed'),
    'pulsed_lfm_10us': dict(kind='pulsed', kw=dict(mod='lfm'), rate=2e6, pulse=True, chirp_pulse=True, group='pulsed'),
    'pulsed_2us_10M': dict(kind='pulsed', kw=dict(pw_s=2e-6, pri_s=1e-3), rate=10e6, pulse=True, group='pulsed'),
    'pulsed_lfm_10us_10M': dict(kind='pulsed', kw=dict(mod='lfm', pw_s=10e-6), rate=10e6, pulse=True, chirp_pulse=True, group='pulsed'),
    # ---- negatives ---------------------------------------------------------------------------------
    'noise': dict(kind='noise', kw={}, rate=2e6, group='noise'),
    'cw': dict(kind='cw', kw={}, rate=2e6, group='tone'),
    'am_tone': dict(kind='am', kw={}, rate=2e6, group='analog'),
    'nfm_tone': dict(kind='nfm', kw=dict(audio='tone'), rate=2e6, group='analog'),
    'nfm_voice': dict(kind='nfm', kw={}, rate=2e6, group='analog'),
    'wfm': dict(kind='wfm', kw={}, rate=2e6, group='analog'),
    'ook_pwm': dict(kind='ook', kw={}, rate=2e6, group='ook'),
    'ook_manchester': dict(kind='ook', kw=dict(coding='manchester'), rate=2e6, group='ook'),
    'ook_nrz_slow': dict(kind='ook', kw=dict(coding='nrz', baud=1000.0), rate=2e6, group='ook'),
    'fsk2_rect': dict(kind='fsk2', kw={}, rate=2e6, group='fsk'),
    'fsk2_gauss': dict(kind='fsk2', kw=dict(shape='gauss'), rate=2e6, group='fsk'),
    'fsk2_slow': dict(kind='fsk2', kw=dict(baud=1200.0, dev_hz=3000.0), rate=2e6, group='fsk'),
    'fsk2_fast': dict(kind='fsk2', kw=dict(baud=50e3, dev_hz=50e3), rate=2e6, group='fsk'),
    'gfsk_250k': dict(kind='gfsk', kw=dict(baud=250e3), rate=2e6, group='fsk'),
    'bpsk': dict(kind='bpsk', kw={}, rate=2e6, group='psk'),
    'qpsk': dict(kind='qpsk', kw={}, rate=2e6, group='psk'),
    'ofdm': dict(kind='ofdm', kw={}, rate=2e6, group='ofdm'),
    'ofdm_narrow': dict(kind='ofdm', kw=dict(bw_hz=300e3), rate=2e6, group='ofdm'),
    'hopper': dict(kind='hopper', kw={}, rate=2e6, group='hopper'),
    'multi_cw_nfm': dict(kind='multi', kw=dict(kinds=('cw', 'nfm')), rate=2e6, group='multi'),
    # ---- mixtures ----------------------------------------------------------------------------------
    'multi_lora_fsk': dict(kind='multi', kw=dict(kinds=('lora', 'fsk2')), rate=2e6, chirp=True, group='mix', target=0),
    'multi_lora_cw': dict(kind='multi', kw=dict(kinds=('lora', 'cw')), rate=2e6, chirp=True, group='mix', target=0),
}

SNRS = (0, 5, 10, 15, 20, 30)
SNRS_LOW = (-12, -9, -6, -3)


def case_params(name, snr, seed):
    """Draw the per-seed impairments.  Returns kwargs for gen.make and the sweep metadata dict."""
    spec = SPECS[name]
    rng = np.random.default_rng([seed, zlib.crc32(name.encode()), int(snr * 10) + 1000])
    err = float(rng.uniform(-60e3, 60e3))                # capture centre vs signal centre (sweep centre error)
    cfo = float(rng.uniform(-5e3, 5e3))
    dc = complex(rng.uniform(-8, 8), rng.uniform(-8, 8))
    onf = float(rng.uniform(0.10, 0.30))
    kw = dict(spec['kw'])
    kind = spec['kind']
    if kind == 'multi':
        kw['rel_offsets_hz'] = (0.0, 400e3)
    if kind == 'noise':
        onf = 1.0
    gk = dict(offset_hz=err, cfo_hz=cfo, dc=dc, on_fraction=onf, **kw)
    return gk, dict(err=err, cfo=cfo, dc=dc, onf=onf), rng


def sweep_meta(truth, rng, target=None, spec=None):
    """What the sweep would have stored: est_bw_hz = number of 100 kHz bins touched by the target signal x 100 kHz
    (optimistic: the sweep sees the whole signal).  For noise: random."""
    if truth['kind'] == 'noise':
        return dict(est_bw_hz=float(rng.choice([100e3, 200e3, 300e3, 500e3])))
    t = truth['components'][target] if (truth['kind'] == 'multi' and target is not None) else truth
    bands = t['bands_hz']
    lo = min(b[0] for b in bands)
    hi = max(b[1] for b in bands)
    ph = float(rng.uniform(0, 100e3))
    nb = int(np.floor((hi - ph) / 100e3) - np.floor((lo - ph) / 100e3) + 1)
    return dict(est_bw_hz=float(100e3 * nb))


def truth_labels(name, truth):
    spec = SPECS[name]
    out = dict(chirp=bool(spec.get('chirp')), pulse=bool(spec.get('pulse')), chirp_pulse=bool(spec.get('chirp_pulse')))
    t = truth
    if truth['kind'] == 'multi' and spec.get('target') is not None:
        t = truth['components'][spec['target']]
    par = t.get('parameters', {})
    out['sf'] = par.get('sf')
    out['bw'] = par.get('bw_hz') if spec['kind'] in ('lora',) or spec.get('target') is not None else None
    if spec['kind'] == 'pulsed':
        pw = par['pw_s']
        out['pw_s'] = pw
        out['pw3db_s'] = 0.932 * pw
        out['pri_s'] = par['pri_s']
        out['jitter'] = par.get('pri_jitter', 0.0)
    out['on_fraction_true'] = float(truth.get('on_fraction', 1.0))
    return out

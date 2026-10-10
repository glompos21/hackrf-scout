"""Scenario definitions (shared by tuning and evaluation).  Seeds 0-9 tune, 100-119 evaluate."""
import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import gen

TUNE_SEEDS = list(range(0, 10))
EVAL_SEEDS = list(range(100, 120))
SNRS = [0, 5, 10, 15, 20, 30]

# variant -> (kind, rate, seconds, kw, truth family for the CLASSIFIER (what a correct answer is))
V = {
    'noise':      ('noise', 2e6, 5.0, {}, 'none'),
    'cw':         ('cw', 2e6, 5.0, {}, 'carrier'),
    'am_tone':    ('am', 2e6, 5.0, dict(audio='tone'), 'am'),
    'am_voice':   ('am', 2e6, 5.0, dict(audio='voice'), 'am'),
    'nfm_voice':  ('nfm', 2e6, 5.0, dict(audio='voice'), 'fm'),
    'nfm_tone':   ('nfm', 2e6, 5.0, dict(audio='tone'), 'fm'),
    'wfm':        ('wfm', 2e6, 5.0, {}, 'fm'),
    'ook_pwm':    ('ook', 2e6, 5.0, {}, 'ook'),
    'ook_man':    ('ook', 2e6, 5.0, dict(coding='manchester', baud=2400.0), 'ook'),
    'ook_nrz':    ('ook', 2e6, 5.0, dict(coding='nrz', baud=2000.0), 'ook'),
    'fsk2_rect':  ('fsk2', 2e6, 5.0, {}, 'fsk'),
    'fsk2_gauss': ('fsk2', 2e6, 5.0, dict(shape='gauss'), 'fsk'),
    'fsk2_h1':    ('fsk2', 2e6, 5.0, dict(baud=40000.0, dev_hz=20000.0), 'fsk'),
    'fsk2_h2':    ('fsk2', 2e6, 5.0, dict(baud=4800.0, dev_hz=9600.0, shape='gauss'), 'fsk'),
    'gfsk':       ('gfsk', 4e6, 5.0, {}, 'fsk'),
    'bpsk':       ('bpsk', 2e6, 5.0, {}, 'psk'),
    'qpsk':       ('qpsk', 2e6, 5.0, {}, 'psk'),
    'qpsk_50k':   ('qpsk', 2e6, 5.0, dict(symrate=50e3), 'psk'),
    'lora':       ('lora', 2e6, 5.0, {}, 'other'),
    'ofdm':       ('ofdm', 10e6, 1.0, {}, 'other'),
    'ofdm_nb':    ('ofdm', 2e6, 5.0, dict(bw_hz=300e3), 'other'),
    'lora_sf9':   ('lora', 2e6, 5.0, dict(sf=9), 'other'),
    'hopper':     ('hopper', 10e6, 1.0, {}, 'other'),
    'pulsed':     ('pulsed', 2e6, 5.0, {}, 'other'),
    'multi_cw_nfm': ('multi', 2e6, 5.0, dict(kinds=('cw', 'nfm')), 'multi'),
    'multi_ook_fsk': ('multi', 2e6, 5.0, dict(kinds=('ook', 'fsk2')), 'multi'),
}
GATED_VARIANTS = ['cw', 'am_tone', 'nfm_voice', 'ook_pwm', 'ook_man', 'fsk2_rect', 'fsk2_gauss', 'gfsk', 'bpsk', 'qpsk', 'wfm']


def impair(variant, snr, seed, cond):
    """Deterministic per-capture impairments (offset, cfo, dc, on_fraction)."""
    h = int(hashlib.md5(('%s|%s|%s|%s' % (variant, snr, seed, cond)).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    kind, rate = V[variant][0], V[variant][1]
    off = float(rng.choice([-1, 1]) * rng.uniform(10e3, 60e3))
    cfo = float(rng.uniform(-5e3, 5e3))
    dcv = float(rng.choice([-1, 1]) * rng.uniform(2, 8))
    dci = float(rng.choice([-1, 1]) * rng.uniform(2, 8))
    onf = 1.0 if cond == 'cont' else float(rng.uniform(0.10, 0.30))
    return dict(offset_hz=off, cfo_hz=cfo, dc=complex(dcv, dci), on_fraction=onf)


def make(variant, snr, seed, cond='cont', seconds=None, return_ci8=False):
    kind, rate, secs, kw, fam = V[variant]
    imp = impair(variant, snr, seed, cond)
    if kind == 'noise':
        imp['on_fraction'] = 1.0
    s = secs if seconds is None else seconds
    # keep multi/gated/ofdm within Nyquist: offsets are small relative to rate/2 for all variants
    return gen.make(kind, rate, s, float(snr), seed=int(seed), return_ci8=return_ci8,
                    offset_hz=imp['offset_hz'], cfo_hz=imp['cfo_hz'], dc=imp['dc'],
                    on_fraction=imp['on_fraction'], **kw), imp


# ---------------------------------------------------------------------------------------------------
# robustness variants (NOT used for tuning or the main evaluation): rate / modulation-index sweeps
# ---------------------------------------------------------------------------------------------------
def _add(name, kind, rate, kw, fam, secs=5.0):
    V[name] = (kind, rate, secs, kw, fam)

for b in (500.0, 1000.0, 10000.0, 20000.0, 50000.0):
    _add('x_ook_pwm_%d' % b, 'ook', 2e6, dict(baud=b), 'ook')
for b, dv in ((1200.0, 3000.0), (2400.0, 4800.0), (9600.0, 19200.0), (19200.0, 25000.0), (50000.0, 50000.0), (100000.0, 50000.0), (200000.0, 100000.0)):
    _add('x_fsk2_%d' % b, 'fsk2', 2e6, dict(baud=b, dev_hz=dv), 'fsk')
_add('x_fsk2_msk', 'fsk2', 2e6, dict(baud=10000.0, dev_hz=2500.0), 'fsk')
_add('x_fsk2_gmsk', 'fsk2', 2e6, dict(baud=10000.0, dev_hz=2500.0, shape='gauss'), 'fsk')
_add('x_fsk2_gfsk_h1_20k', 'fsk2', 2e6, dict(baud=20000.0, dev_hz=10000.0, shape='gauss'), 'fsk')
for sr in (10e3, 100e3, 500e3):
    _add('x_qpsk_%d' % sr, 'qpsk', 2e6, dict(symrate=sr), 'psk')
_add('x_bpsk_100k', 'bpsk', 2e6, dict(symrate=100e3), 'psk')
_add('x_fsk2_10M', 'fsk2', 10e6, dict(), 'fsk', 2.0)
_add('x_ook_10M', 'ook', 10e6, dict(), 'ook', 2.0)
_add('x_qpsk_10M', 'qpsk', 10e6, dict(symrate=1e6), 'psk', 2.0)
_add('x_nfm_10M', 'nfm', 10e6, dict(), 'fm', 2.0)

#!/usr/bin/env python3
"""Synthetic IQ test bench for the hackrf-scout "characterise captured IQ" study.

numpy only (Python >= 3.9).  One entry point:

    res = make(kind, rate, seconds, snr_db, seed=0, offset_hz=0.0, on_fraction=1.0,
               dc=0.0, cfo_hz=0.0, rms_lsb=20.0, return_ci8=True, **kw)

    res['iq']    complex64, the *quantised* signal as the receiver sees it (I/Q in LSB, integers
                 in [-128, 127] stored as float32; exactly what hackrf_transfer's ci8 holds)
    res['ci8']   bytes, interleaved int8 I,Q,I,Q... (None when return_ci8=False)
    res['truth'] dict (see below)

SNR DEFINITION (state it, test it)
    snr_db is the IN-BAND, ON-STATE SNR:

        snr = P_on / (N0 * B)        N0 = noise PSD [power/Hz], white over the WHOLE sampled bandwidth
                                     B  = truth['occupied_bw_hz'] (the signal's occupied bandwidth)
                                     P_on = mean signal power over the RF-on intervals only
                                            (truth['on_intervals']); gaps/off-chips do not dilute it

    It is therefore comparable with a sweep bin power over a signal-sized bandwidth, and it does not
    depend on the sample rate (a narrowband signal at 2 or 10 Msps with the same snr_db has the same
    in-band SNR; its *wideband* SNR is lower by 10*log10(B/rate)).
    Quantisation: (signal + noise) is scaled so that the per-component RMS of the I/Q stream
    (sqrt(mean(I^2+Q^2)/2)) equals rms_lsb, THEN dc is added, THEN rounded and clipped to int8.
    Hence a strong, low-duty or very wideband signal can clip: see truth['clip_fraction'] and
    truth['warnings'].

TRUTH  (res['truth'])
    kind, family, rate, seconds, n, seed, snr_db, offset_hz, cfo_hz, center_hz (= offset_hz + cfo_hz),
    dc_lsb, rms_lsb, noise_rms_lsb (per I or Q component, after scaling), noise_psd_lsb2_per_hz (complex power/Hz),
    clip_fraction (share of samples with I or Q beyond the int8 range before clipping), warnings, snr_definition,
    occupied_bw_hz  B of the SNR definition (None for 'noise'; override with occ_bw_hz=...),
    bands_hz        absolute (lo, hi) spectral extents of the signal; hopper: one per channel (occupied_bw_hz is then
                    the instantaneous per-hop bandwidth); the nominal B holds >= 97 % of the power (selftest),
    symbol_rate     effective keying/chip/symbol/baud rate in Hz or None (lora: BW/2^SF; ofdm: 1/symbol time),
    bursts          [(start_s, end_s)] frame / packet / pulse / hop extents (base to base, incl. ramps),
    on_intervals    [(start_s, end_s)] where RF is actually on (OOK: the keyed-on chips; otherwise == bursts),
    windows, gate_fraction   the random gate windows implied by on_fraction (bursts lie inside them),
    on_fraction     on-time / capture length, on_power_lsb2 (P_on after scaling), parameters (effective kw + derived),
    hop_freqs_hz    (hopper) absolute carrier of each burst, components (multi only: one truth dict per component).

Everything is deterministic given (all arguments, seed).  Unsupported combinations raise ValueError / TypeError.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

TWO_PI = 2.0 * math.pi
_BLK = 1 << 20  # processing block (samples) so 5 s @ 10 Msps never needs giant temporaries

KINDS = ('noise', 'cw', 'am', 'nfm', 'wfm', 'ook', 'fsk2', 'gfsk', 'bpsk', 'qpsk',
         'lora', 'ofdm', 'hopper', 'pulsed', 'multi')

# Coarse family label that a characteriser could be expected to output (ground truth hint only).
FAMILY = {
    'noise': 'noise', 'cw': 'carrier', 'am': 'am', 'nfm': 'fm_analog', 'wfm': 'fm_analog',
    'ook': 'ook', 'fsk2': 'fsk', 'gfsk': 'fsk', 'bpsk': 'psk', 'qpsk': 'psk',
    'lora': 'chirp', 'ofdm': 'ofdm', 'hopper': 'hopper', 'pulsed': 'pulsed', 'multi': 'multi',
}

# kw defaults per kind (every kind additionally accepts occ_bw_hz=None to override the SNR bandwidth B)
_DEF = {
    'noise': {},
    'cw': dict(bw_hz=1000.0, linewidth_hz=5.0, drift_hz_per_s=0.0),
    'am': dict(audio='tone', tone_hz=1000.0, depth=0.8, audio_hi_hz=3000.0),
    'nfm': dict(audio='voice', tone_hz=1000.0, dev_hz=2500.0, audio_hi_hz=3000.0, bw_hz=12500.0),
    'wfm': dict(dev_hz=75000.0, programme_rms=0.32, stereo=True, pilot=0.09, bw_hz=200000.0),
    'ook': dict(baud=3000.0, coding='pwm', n_bits=32, preamble_pairs=8, frame_gap_s=0.015,
                depth=1.0, edge=0.35),
    'fsk2': dict(baud=10000.0, dev_hz=25000.0, shape='rect', bt=0.5),
    'gfsk': dict(baud=1.0e6, h=0.5, bt=0.5, payload_bytes=(4, 37), gap_s=(150e-6, 2e-3)),
    'bpsk': dict(symrate=250e3, rolloff=0.35, span=10),
    'qpsk': dict(symrate=250e3, rolloff=0.35, span=10),
    'lora': dict(sf=7, bw_hz=125e3, n_preamble=8, n_payload=(8, 24), gap_s=(20e-3, 80e-3)),
    'ofdm': dict(bw_hz=None, mod='qpsk', n_sym=(8, 120), gap_s=(20e-6, 400e-6)),
    'hopper': dict(n_channels=5, baud=None, chan_spacing_hz=None, dwell_s=625e-6, duty=0.85,
                   h=0.5, bt=0.5, ramp_s=8e-6),
    'pulsed': dict(pw_s=10e-6, pri_s=1e-3, pri_jitter=0.0, mod='none', chirp_bw_hz=None, rise=0.25),
    'multi': dict(kinds=('cw', 'nfm'), rel_offsets_hz=None, snrs_db=None, on_fractions=None,
                  sub_kw=None),
}

# nominal occupied-bandwidth factors (calibrated by selftest(): >= ~97 % of the power inside B)
_BW_OOK = 4.0            # x chip rate
_BW_FSK_RECT = 1.5       # 2*(dev + k*baud)
_BW_FSK_GAUSS = 1.0
_BW_GFSK = 1.2           # x baud
_BW_LORA = 1.0           # x chirp bandwidth
_BW_OFDM_EXTRA = 3.0     # B = df*(52 + extra)
_BW_PULSE = 5.0          # B = k / pw  (+ chirp bw)


# ----------------------------------------------------------------------------------------------
# low level helpers
# ----------------------------------------------------------------------------------------------
def _cis(phase):
    """exp(j*phase) as complex64; phase is reduced mod 2*pi in float64 first."""
    p = np.remainder(phase, TWO_PI).astype(np.float32)
    out = np.empty(p.shape, np.complex64)
    np.cos(p, out=out.real)
    np.sin(p, out=out.imag)
    return out


def _lerp(arr, x):
    """Linear interpolation of 1-D float32 arr at fractional index array x (clamped)."""
    n = len(arr)
    x = np.clip(x, 0.0, n - 1.0)
    i0 = np.minimum(x.astype(np.int64), n - 2)
    f = (x - i0).astype(np.float32)
    return arr[i0] * (1.0 - f) + arr[i0 + 1] * f


def _sstep(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def _fm_synth(freq_block, n, rate, phi0=0.0):
    """complex64 exp(j*2pi*int f dt); freq_block(a, b) -> frequency in Hz for samples a..b."""
    out = np.empty(n, np.complex64)
    ph = float(phi0)
    for a in range(0, n, _BLK):
        b = min(n, a + _BLK)
        f = np.asarray(freq_block(a, b), dtype=np.float64)
        inc = np.cumsum(f) * (TWO_PI / rate)
        out[a:b] = _cis(ph + inc)
        ph = math.fmod(ph + float(inc[-1]), TWO_PI)
    return out


def _pulse_poly(a, g, sps):
    """y[i] = sum_k a[k] * g[i - k*sps] (polyphase: cost n_sym * len(g) not n * len(g))."""
    L = -(-len(g) // sps)
    gp = np.zeros(L * sps, np.float32)
    gp[:len(g)] = g
    G = gp.reshape(L, sps)
    m = len(a) + L - 1
    out = np.empty((m, sps), a.dtype)
    for p in range(sps):
        out[:, p] = np.convolve(a, G[:, p])
    return out.reshape(-1)


def _pick_sps(rate, baud, internal=16):
    """(sps, exact): use the true integer samples/symbol when 4 <= rate/baud <= 32, else an
    internal oversampled rate (internal*baud) that is linearly interpolated to `rate`."""
    r = rate / baud
    if abs(r - round(r)) < 1e-6 * r and 4 <= round(r) <= 32:
        return int(round(r)), True
    return internal, False


def _nrz_shaped(levels, baud, rate, g, delay, sps, exact, lead=None, trail=None, pad=3):
    """Pulse-shape a per-symbol level sequence; symbol k spans [k/baud, (k+1)/baud) in output time.
    Returns blk(a, b) -> float32 waveform at output samples a..b."""
    lv = np.concatenate([np.full(pad, levels[0] if lead is None else lead, np.float32),
                         np.asarray(levels, np.float32),
                         np.full(pad, levels[-1] if trail is None else trail, np.float32)])
    y = _pulse_poly(lv, np.asarray(g, np.float32), sps)
    off = delay + pad * sps
    if exact:
        o = int(round(off))

        def blk(a, b):
            seg = y[a + o:b + o]
            if len(seg) < b - a:
                seg = np.pad(seg, (0, b - a - len(seg)), mode='edge')
            return seg
    else:
        ratio = baud * sps / rate

        def blk(a, b):
            return _lerp(y, np.arange(a, b, dtype=np.float64) * ratio + off)
    return blk


def _box_hann_pulse(sps, edge):
    """NRZ pulse with raised-cosine edges: box(sps) convolved with a Hann kernel of edge*sps taps."""
    k = max(1, int(round(edge * sps)))
    ker = np.hanning(k + 2)[1:-1] if k > 1 else np.ones(1)
    ker = ker / ker.sum()
    g = np.convolve(np.ones(sps), ker)
    return g.astype(np.float32), (len(ker) - 1) / 2.0


def _box_gauss_pulse(sps, bt):
    """Gaussian-filtered NRZ pulse (BT = bandwidth-bit-time product)."""
    sigma = math.sqrt(math.log(2.0)) / (TWO_PI * bt) * sps
    hw = max(1, int(math.ceil(3.5 * sigma)))
    x = np.arange(-hw, hw + 1)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= ker.sum()
    g = np.convolve(np.ones(sps), ker)
    return g.astype(np.float32), float(hw)


def _rrc(beta, sps, span):
    n = span * sps
    t = np.arange(-(n // 2), n // 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / math.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-9:
            h[i] = (beta / math.sqrt(2.0)) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * beta))
                                              + (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta)))
        else:
            h[i] = ((math.sin(math.pi * ti * (1 - beta)) + 4 * beta * ti * math.cos(math.pi * ti * (1 + beta)))
                    / (math.pi * ti * (1 - (4 * beta * ti) ** 2)))
    return (h / math.sqrt((h ** 2).sum())).astype(np.float32)


def _ramp_edges(w, nr):
    """raised-cosine fade in/out over nr samples (in place)."""
    nr = int(min(nr, len(w) // 2))
    if nr < 1:
        return w
    r = (0.5 - 0.5 * np.cos(np.pi * (np.arange(nr) + 0.5) / nr)).astype(np.float32)
    w[:nr] *= r
    w[len(w) - nr:] *= r[::-1]
    return w


def _bandnoise(rng, n, fs, f_lo, f_hi):
    """unit-variance Gaussian noise band-limited to [f_lo, f_hi] (FFT shaped, soft edges)."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    w = max(0.1 * (f_hi - f_lo), 3.0 * fs / n)
    H = _sstep((f - f_lo) / w + 0.5) * _sstep((f_hi - f) / w + 0.5)
    y = np.fft.irfft(X * H, n)
    s = y.std()
    return (y / s if s > 0 else y).astype(np.float32)


def _voice(rng, n, fs, f_lo=300.0, f_hi=3000.0, rms_target=0.35):
    """voice-like audio in [-1, 1]: band-limited noise with a syllabic (~4 Hz) on/off envelope."""
    base = _bandnoise(rng, n, fs, f_lo, f_hi)
    slow = _bandnoise(rng, n, fs, 0.3, 5.0)
    env = np.clip((slow + 0.6) / 1.4, 0.0, 1.0)
    v = base * env
    rms = float(np.sqrt(np.mean(v ** 2))) or 1.0
    return np.clip(v * (rms_target / rms), -1.0, 1.0).astype(np.float32)


def _qam(rng, shape, mod):
    if mod == 'bpsk':
        return (2.0 * rng.integers(0, 2, shape) - 1.0).astype(np.complex64)
    if mod == 'qpsk':
        q = 2.0 * rng.integers(0, 2, shape + (2,)) - 1.0
        return ((q[..., 0] + 1j * q[..., 1]) / math.sqrt(2.0)).astype(np.complex64)
    if mod == '16qam':
        q = (2 * rng.integers(0, 4, shape + (2,)) - 3).astype(np.float64)
        return ((q[..., 0] + 1j * q[..., 1]) / math.sqrt(10.0)).astype(np.complex64)
    raise ValueError("mod must be 'bpsk', 'qpsk' or '16qam'")


def _runs(mask01):
    d = np.diff(np.concatenate([[0], np.asarray(mask01, np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _train(rng, dur, rate, make_pkt, gap_range_s):
    """Fill [0,dur) with packets separated by random gaps.  make_pkt(max_len) -> wave or None."""
    wave = np.zeros(dur, np.complex64)
    bursts = []
    glo, ghi = gap_range_s
    pos = int(rng.uniform(0.0, ghi) * rate)
    while pos < dur:
        pkt = make_pkt(dur - pos)
        if pkt is None:
            break
        wave[pos:pos + len(pkt)] = pkt
        bursts.append((pos, pos + len(pkt)))
        pos += len(pkt) + max(1, int(rng.uniform(glo, ghi) * rate))
    if not bursts and dur >= 8:      # window too small for random lead-in: force one packet at 0
        pkt = make_pkt(dur)
        if pkt is not None:
            wave[:len(pkt)] = pkt
            bursts.append((0, len(pkt)))
    return wave, bursts, list(bursts)


def _full(dur):
    return [(0, dur)]


# ----------------------------------------------------------------------------------------------
# kinds.  Each _k_<kind>(rate, seconds, p) returns a spec dict:
#   bw (occupied bw B, Hz), symrate (Hz or None), bands (relative (lo,hi) Hz list), params,
#   min_len / typ_len (samples), gen(rng, dur) -> (wave complex64[dur], bursts, on)   [sample units]
# ----------------------------------------------------------------------------------------------
def _cont_lens(rate, seconds):
    return max(256, int(0.005 * rate)), max(int(0.05 * rate), int(rate * seconds / 8))


def _k_noise(rate, seconds, p):
    return dict(bw=None, symrate=None, bands=[], params={}, min_len=1, typ_len=1, gen=None)


def _k_cw(rate, seconds, p):
    lw, dr = float(p['linewidth_hz']), float(p['drift_hz_per_s'])

    def gen(rng, dur):
        out = np.empty(dur, np.complex64)
        rw = 0.0
        sd = math.sqrt(TWO_PI * lw / rate) if lw > 0 else 0.0
        for a in range(0, dur, _BLK):
            b = min(dur, a + _BLK)
            if sd:
                inc = np.cumsum(rng.standard_normal(b - a)) * sd + rw
                rw = float(inc[-1])
            else:
                inc = np.zeros(b - a)
            if dr:
                t = np.arange(a, b, dtype=np.float64) / rate
                inc = inc + math.pi * dr * t * t
            out[a:b] = _cis(inc)
        return out, _full(dur), _full(dur)

    mn, ty = _cont_lens(rate, seconds)
    B = float(p['bw_hz'])
    return dict(bw=B, symrate=None, bands=[(-B / 2, B / 2)], params=dict(p), min_len=mn, typ_len=ty, gen=gen)


def _k_am(rate, seconds, p):
    if p['audio'] not in ('tone', 'voice'):
        raise ValueError("am: audio must be 'tone' or 'voice'")
    depth = float(p['depth'])
    fm = 48e3
    fmax = float(p['tone_hz'] if p['audio'] == 'tone' else p['audio_hi_hz'])
    B = 2.0 * fmax + 500.0

    def gen(rng, dur):
        out = np.empty(dur, np.complex64)
        if p['audio'] == 'tone':
            ph0 = rng.uniform(0, TWO_PI)
            for a in range(0, dur, _BLK):
                b = min(dur, a + _BLK)
                cyc = np.remainder(np.arange(a, b, dtype=np.float64) * (p['tone_hz'] / rate), 1.0)
                out[a:b] = (1.0 + depth * np.cos(TWO_PI * cyc + ph0)).astype(np.float32)
        else:
            na = int(math.ceil(dur * fm / rate)) + 4
            aud = _voice(rng, na, fm, 300.0, float(p['audio_hi_hz']))
            for a in range(0, dur, _BLK):
                b = min(dur, a + _BLK)
                x = np.arange(a, b, dtype=np.float64) * (fm / rate)
                out[a:b] = 1.0 + depth * _lerp(aud, x)
        return out, _full(dur), _full(dur)

    mn, ty = _cont_lens(rate, seconds)
    return dict(bw=B, symrate=None, bands=[(-B / 2, B / 2)], params=dict(p), min_len=max(mn, int(0.02 * rate)),
                typ_len=ty, gen=gen)


def _k_nfm(rate, seconds, p):
    if p['audio'] not in ('tone', 'voice'):
        raise ValueError("nfm: audio must be 'tone' or 'voice'")
    dev = float(p['dev_hz'])
    fm = 48e3
    B = float(p['bw_hz'])

    def gen(rng, dur):
        na = int(math.ceil(dur * fm / rate)) + 4
        if p['audio'] == 'tone':
            msg = np.sin(TWO_PI * p['tone_hz'] * np.arange(na) / fm + rng.uniform(0, TWO_PI)).astype(np.float32)
        else:
            msg = _voice(rng, na, fm, 300.0, float(p['audio_hi_hz']), 0.45)
        fb = lambda a, b: dev * _lerp(msg, np.arange(a, b, dtype=np.float64) * (fm / rate))
        return _fm_synth(fb, dur, rate, rng.uniform(0, TWO_PI)), _full(dur), _full(dur)

    mn, ty = _cont_lens(rate, seconds)
    return dict(bw=B, symrate=None, bands=[(-B / 2, B / 2)], params=dict(p), min_len=max(mn, int(0.02 * rate)),
                typ_len=ty, gen=gen)


def _k_wfm(rate, seconds, p):
    B = float(p['bw_hz'])
    if rate < 2.5 * B:
        raise ValueError('wfm: needs rate >= %.0f sps' % (2.5 * B))
    dev, fm = float(p['dev_hz']), 380e3
    prog, pilot, stereo = float(p['programme_rms']), float(p['pilot']), bool(p['stereo'])

    def gen(rng, dur):
        na = int(math.ceil(dur * fm / rate)) + 4
        t = np.arange(na, dtype=np.float64) / fm
        common = _bandnoise(rng, na, fm, 50.0, 15e3)
        env = np.clip((_bandnoise(rng, na, fm, 0.2, 3.0) + 1.5) / 2.5, 0.2, 1.0)
        if stereo:
            lch = (0.8 * common + 0.6 * _bandnoise(rng, na, fm, 50.0, 15e3)) * env
            rch = (0.8 * common + 0.6 * _bandnoise(rng, na, fm, 50.0, 15e3)) * env
            php = rng.uniform(0, TWO_PI)
            programme = (0.5 * (lch + rch)
                         + 0.5 * (lch - rch) * np.sin(2 * TWO_PI * 19e3 * t + 2 * php)).astype(np.float32)
        else:
            programme = (common * env).astype(np.float32)
            php = 0.0
        r = float(np.sqrt(np.mean(programme ** 2))) or 1.0
        mpx = np.clip(programme * (prog / r), -(1.0 - pilot), 1.0 - pilot)
        if stereo and pilot > 0:
            mpx = mpx + pilot * np.sin(TWO_PI * 19e3 * t + php).astype(np.float32)
        mpx = mpx.astype(np.float32)
        fb = lambda a, b: dev * _lerp(mpx, np.arange(a, b, dtype=np.float64) * (fm / rate))
        return _fm_synth(fb, dur, rate, rng.uniform(0, TWO_PI)), _full(dur), _full(dur)

    mn, ty = _cont_lens(rate, seconds)
    return dict(bw=B, symrate=None, bands=[(-B / 2, B / 2)], params=dict(p), min_len=max(mn, int(0.02 * rate)),
                typ_len=ty, gen=gen)


def _ook_chips(rng, p):
    coding, nb = p['coding'], int(p['n_bits'])
    pre = [1, 0] * int(p['preamble_pairs'])
    bits = rng.integers(0, 2, nb)
    if coding == 'pwm':
        sync = [1, 1, 1, 1] + [0] * 8
        body = np.concatenate([[1, 1, 0] if b else [1, 0, 0] for b in bits])
    elif coding == 'manchester':
        sync = [1, 1, 0, 0]
        body = np.concatenate([[1, 0] if b else [0, 1] for b in bits])
    elif coding == 'nrz':
        sync = [1, 1, 1, 0]
        body = bits
    else:
        raise ValueError("ook: coding must be 'pwm', 'manchester' or 'nrz'")
    return np.concatenate([[0], pre, sync, body, [0, 0]]).astype(np.int8)  # leading/trailing off chip


def _k_ook(rate, seconds, p):
    baud = float(p['baud'])
    if rate < 8 * baud:
        raise ValueError('ook: needs rate >= 8*baud')
    depth, edge = float(p['depth']), float(p['edge'])
    sps, exact = _pick_sps(rate, baud, internal=32)
    g, delay = _box_hann_pulse(sps, edge)
    chips_per_bit = {'pwm': 3, 'manchester': 2, 'nrz': 1}[p['coding']]
    n_chips = 1 + 2 * int(p['preamble_pairs']) + {'pwm': 12, 'manchester': 4, 'nrz': 4}[p['coding']] \
        + chips_per_bit * int(p['n_bits']) + 2
    chip_s = rate / baud
    frame_len = int(math.ceil(n_chips * chip_s))
    gap = float(p['frame_gap_s']) * rate

    def gen(rng, dur):
        chips = _ook_chips(rng, p)
        lv = (1.0 - depth) + depth * chips.astype(np.float32)
        blk = _nrz_shaped(lv, baud, rate, g, delay, sps, exact, lead=lv[0], trail=lv[-1])
        frame = blk(0, frame_len).astype(np.complex64)
        st, en = _runs(chips)
        on_rel = [(int(round(s * chip_s)), int(round(e * chip_s))) for s, e in zip(st, en)]
        wave = np.zeros(dur, np.complex64)
        bursts, on = [], []
        pos = int(rng.uniform(0, gap))
        while pos + frame_len <= dur:
            wave[pos:pos + frame_len] = frame
            bursts.append((pos + on_rel[0][0], pos + on_rel[-1][1]))
            on += [(pos + s, pos + e) for s, e in on_rel]
            pos += frame_len + int(gap * rng.uniform(0.9, 1.1))
        if not bursts:
            raise ValueError('ook: window shorter than one frame')
        return wave, bursts, on

    pr = dict(p)
    pr['bit_rate'] = baud / chips_per_bit
    pr['frame_s'] = frame_len / rate
    return dict(bw=_BW_OOK * baud, symrate=baud, bands=[(-_BW_OOK * baud / 2, _BW_OOK * baud / 2)],
                params=pr, min_len=frame_len + int(gap) + 8, typ_len=3 * (frame_len + int(gap)), gen=gen)


def _k_fsk2(rate, seconds, p):
    baud, dev, shape = float(p['baud']), float(p['dev_hz']), p['shape']
    if shape not in ('rect', 'gauss'):
        raise ValueError("fsk2: shape must be 'rect' or 'gauss'")
    if rate < 4 * baud:
        raise ValueError('fsk2: needs rate >= 4*baud')
    k = _BW_FSK_RECT if shape == 'rect' else _BW_FSK_GAUSS
    B = 2.0 * (dev + k * baud)
    if shape == 'gauss':
        sps, exact = _pick_sps(rate, baud, internal=16)
        g, delay = _box_gauss_pulse(sps, float(p['bt']))

    def gen(rng, dur):
        nsym = int(math.ceil(dur * baud / rate)) + 2
        lv = (2.0 * rng.integers(0, 2, nsym) - 1.0).astype(np.float32)
        if shape == 'rect':
            def fb(a, b):
                idx = np.minimum((np.arange(a, b, dtype=np.float64) * (baud / rate)).astype(np.int64), nsym - 1)
                return dev * lv[idx]
        else:
            blk = _nrz_shaped(lv, baud, rate, g, delay, sps, exact)
            fb = lambda a, b: dev * blk(a, b)
        return _fm_synth(fb, dur, rate, rng.uniform(0, TWO_PI)), _full(dur), _full(dur)

    mn, ty = _cont_lens(rate, seconds)
    pr = dict(p)
    pr['mod_index'] = 2 * dev / baud
    return dict(bw=B, symrate=baud, bands=[(-B / 2, B / 2)], params=pr, min_len=mn, typ_len=ty, gen=gen)


def _gfsk_stream(rng, n, rate, baud, h, bt, phi0=0.0):
    nsym = int(math.ceil(n * baud / rate)) + 2
    lv = (2.0 * rng.integers(0, 2, nsym) - 1.0).astype(np.float32)
    sps, exact = _pick_sps(rate, baud, internal=16)
    g, delay = _box_gauss_pulse(sps, bt)
    blk = _nrz_shaped(lv, baud, rate, g, delay, sps, exact)
    dev = h * baud / 2.0
    return _fm_synth(lambda a, b: dev * blk(a, b), n, rate, phi0)


def _k_gfsk(rate, seconds, p):
    baud, h, bt = float(p['baud']), float(p['h']), float(p['bt'])
    if rate < 4 * baud:
        raise ValueError('gfsk: needs rate >= 4*baud (>= 4 Msps for 1 Mbaud)')
    B = _BW_GFSK * baud
    pb = p['payload_bytes']
    pb = (int(pb), int(pb)) if np.isscalar(pb) else (int(pb[0]), int(pb[1]))
    gap_rng = tuple(float(x) for x in p['gap_s'])
    fixed_bits = 8 + 32 + 24
    bit_s = rate / baud
    ramp = max(2, int(round(2e-6 * rate)))

    def gen(rng, dur):
        def mk(max_len):
            nmax = int((max_len / bit_s - fixed_bits) // 8)
            if nmax < pb[0]:
                return None
            nb = int(rng.integers(pb[0], min(pb[1], nmax) + 1))
            nbits = fixed_bits + 8 * nb
            n = int(round(nbits * bit_s))
            w = _gfsk_stream(rng, n, rate, baud, h, bt, rng.uniform(0, TWO_PI))
            return _ramp_edges(w, ramp)
        return _train(rng, dur, rate, mk, gap_rng)

    minlen = int((fixed_bits + 8 * pb[0]) * bit_s * 1.2) + 8
    pr = dict(p)
    pr['mod_index'] = h
    pr['dev_hz'] = h * baud / 2
    return dict(bw=B, symrate=baud, bands=[(-B / 2, B / 2)], params=pr, min_len=minlen,
                typ_len=max(int(5e-3 * rate), 3 * minlen), gen=gen)


def _k_psk(kind):
    def f(rate, seconds, p):
        sr, beta, span = float(p['symrate']), float(p['rolloff']), int(p['span'])
        sps = int(round(rate / sr))
        if sps < 2:
            raise ValueError('%s: needs rate >= 2*symrate' % kind)
        sr_eff = rate / sps
        h = _rrc(beta, sps, span)
        c = (len(h) - 1) // 2
        B = (1.0 + beta) * sr_eff

        def gen(rng, dur):
            nsym = int(math.ceil(dur / sps)) + 1
            a = _qam(rng, (nsym,), kind)
            y = _pulse_poly(a, h, sps)
            return np.ascontiguousarray(y[c:c + dur]), _full(dur), _full(dur)

        mn, ty = _cont_lens(rate, seconds)
        pr = dict(p)
        pr['symrate_effective'] = sr_eff
        return dict(bw=B, symrate=sr_eff, bands=[(-B / 2, B / 2)], params=pr, min_len=max(mn, 4 * span * sps),
                    typ_len=ty, gen=gen)
    return f


def _lora_wave(rate, sf, bw, slots, quarter):
    """slots: list of (is_down, k).  The last slot is only 0.25 symbol long when quarter=True."""
    M = 1 << sf
    T = M / bw
    nslot = len(slots)
    L = int(round((nslot - 1 + (0.25 if quarter else 1.0)) * T * rate))
    t = np.arange(L, dtype=np.float64) / rate
    m = np.minimum((t / T).astype(np.int64), nslot - 1)
    tau = t - m * T
    down = np.array([s[0] for s in slots], bool)[m]
    k = np.array([s[1] for s in slots], np.float64)[m]
    tw = (M - k) / bw
    up = (-bw / 2 + k * bw / M) * tau + (bw / (2 * T)) * tau * tau - bw * (tau - tw) * (tau > tw)
    dn = -((-bw / 2) * tau + (bw / (2 * T)) * tau * tau)
    return _cis(TWO_PI * np.where(down, dn, up))


def _k_lora(rate, seconds, p):
    sf, bw = int(p['sf']), float(p['bw_hz'])
    if not 5 <= sf <= 12:
        raise ValueError('lora: sf must be 5..12')
    if rate < 2 * bw:
        raise ValueError('lora: needs rate >= 2*bw_hz')
    M = 1 << sf
    T = M / bw
    npre = int(p['n_preamble'])
    npay = p['n_payload']
    npay = (int(npay), int(npay)) if np.isscalar(npay) else (int(npay[0]), int(npay[1]))
    gap_rng = tuple(float(x) for x in p['gap_s'])
    ramp = max(2, int(round(5e-6 * rate)))
    B = _BW_LORA * bw
    head = npre + 2 + 2.25                      # preamble + 2 sync + SFD (symbols)

    def gen(rng, dur):
        def mk(max_len):
            nmax = int(max_len / (T * rate) - head)
            if nmax < npay[0]:
                return None
            npy = int(rng.integers(npay[0], min(npay[1], nmax) + 1))
            slots = ([(False, 0)] * npre + [(False, 8 % M), (False, 16 % M)] + [(True, 0), (True, 0), (True, 0)]
                     + [(False, int(k)) for k in rng.integers(0, M, npy)])
            # SFD quarter chirp sits between the down chirps and the payload: build in two parts
            pre = _lora_wave(rate, sf, bw, slots[:npre + 2 + 3], True)
            pay = _lora_wave(rate, sf, bw, slots[npre + 2 + 3:], False)
            w = np.concatenate([pre, pay])
            return _ramp_edges(w, ramp)
        return _train(rng, dur, rate, mk, gap_rng)

    minlen = int((head + npay[0]) * T * rate * 1.05) + 8
    pr = dict(p)
    pr['chirp_time_s'] = T
    pr['chip_rate'] = bw
    return dict(bw=B, symrate=bw / M, bands=[(-B / 2, B / 2)], params=pr, min_len=minlen,
                typ_len=max(3 * minlen, int(0.1 * rate)), gen=gen)


def _ofdm_setup(N, mod_rng_seed=12345):
    r = np.random.default_rng(mod_rng_seed)
    used = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
    pil = np.array([-21, -7, 7, 21])
    data = np.array([k for k in used if k not in pil])
    sts_k = np.array([k for k in range(-24, 25, 4) if k != 0])
    X = np.zeros(N, np.complex128)
    X[sts_k % N] = (2 * r.integers(0, 2, len(sts_k)) - 1) + 1j * (2 * r.integers(0, 2, len(sts_k)) - 1)
    sts = np.fft.ifft(X)
    sts = sts / np.sqrt(np.mean(np.abs(sts) ** 2))
    X = np.zeros(N, np.complex128)
    X[used % N] = 2 * r.integers(0, 2, len(used)) - 1
    lts = np.fft.ifft(X)
    lts = lts / np.sqrt(np.mean(np.abs(lts) ** 2))
    preamble = np.concatenate([np.tile(sts, 3)[:int(2.5 * N)], lts[-N // 2:], lts, lts]).astype(np.complex64)
    return used, pil, data, preamble


def _k_ofdm(rate, seconds, p):
    bw_req = p['bw_hz'] if p['bw_hz'] is not None else min(0.8 * rate, 16.6e6)
    bw_req = float(bw_req)
    if bw_req > 0.8 * rate * 1.0001:
        raise ValueError('ofdm: bw_hz must be <= 0.8*rate')
    N = max(64, 4 * int(math.ceil(52.0 * rate / (4.0 * bw_req))))
    df = rate / N
    Ncp = N // 4
    L = N + Ncp
    w = max(2, N // 8)
    B = df * (52 + _BW_OFDM_EXTRA)
    used, pil, data_k, preamble = _ofdm_setup(N)
    ns = p['n_sym']
    ns = (int(ns), int(ns)) if np.isscalar(ns) else (int(ns[0]), int(ns[1]))
    gap_rng = tuple(float(x) for x in p['gap_s'])
    mod = p['mod']
    _qam(np.random.default_rng(0), (1,), mod)  # validate
    ramp_up = (_sstep((np.arange(w) + 0.5) / w)).astype(np.float32)
    pol = (2 * np.random.default_rng(7).integers(0, 2, 127) - 1)
    pre_len = len(preamble)

    def packet(rng, nsym):
        X = np.zeros((nsym, N), np.complex64)
        X[:, data_k % N] = _qam(rng, (nsym, len(data_k)), mod)
        pv = np.array([1, 1, 1, -1]) * pol[np.arange(nsym) % 127][:, None]
        X[:, pil % N] = pv
        x = np.fft.ifft(X, axis=1).astype(np.complex64) * np.float32(N / math.sqrt(52.0))
        sym = np.concatenate([x[:, -Ncp:], x], axis=1)            # cyclic prefix
        ext = np.concatenate([sym, x[:, :w]], axis=1)              # cyclic extension for the overlap window
        ext[:, :w] *= ramp_up
        ext[:, L:] *= ramp_up[::-1]
        out = np.zeros(((nsym + 1) * L,), np.complex64).reshape(nsym + 1, L)
        out[:nsym] += ext[:, :L]
        out[1:, :w] += ext[:, L:]
        flat = out.reshape(-1)[:nsym * L + w]
        pkt = np.concatenate([preamble, flat])
        return _ramp_edges(pkt, w)

    def gen(rng, dur):
        def mk(max_len):
            nmax = (max_len - pre_len - w) // L
            if nmax < ns[0]:
                return None
            return packet(rng, int(rng.integers(ns[0], min(ns[1], nmax) + 1)))
        return _train(rng, dur, rate, mk, gap_rng)

    minlen = pre_len + ns[0] * L + w + 8
    pr = dict(p)
    pr.update(n_fft=N, subcarrier_spacing_hz=df, cp_samples=Ncp, symbol_time_s=L / rate, bw_hz_effective=52 * df)
    return dict(bw=B, symrate=rate / L, bands=[(-B / 2, B / 2)], params=pr, min_len=int(minlen * 1.1),
                typ_len=max(3 * minlen, int(3e-3 * rate)), gen=gen)


def _k_hopper(rate, seconds, p):
    nch = int(p['n_channels'])
    baud = float(p['baud']) if p['baud'] else min(1.0e6, rate / 8.0)
    spacing = float(p['chan_spacing_hz']) if p['chan_spacing_hz'] else baud
    h, bt = float(p['h']), float(p['bt'])
    dwell = float(p['dwell_s']) * rate
    duty = float(p['duty'])
    if nch < 2:
        raise ValueError('hopper: n_channels >= 2')
    if rate < 4 * baud:
        raise ValueError('hopper: needs rate >= 4*baud')
    B = _BW_GFSK * baud
    cf = (np.arange(nch) - (nch - 1) / 2.0) * spacing
    if (cf.max() + B / 2) * 2 >= rate:
        raise ValueError('hopper: channel set does not fit in the sampled bandwidth')
    ramp = max(2, int(round(float(p['ramp_s']) * rate)))

    def gen(rng, dur):
        g = _gfsk_stream(rng, dur, rate, baud, h, bt, rng.uniform(0, TWO_PI))
        wave = np.zeros(dur, np.complex64)
        bursts = []
        nslot = int(dur // dwell) + 1
        order = []
        prev = -1
        while len(order) < nslot:
            perm = list(rng.permutation(nch))
            if perm[0] == prev:
                perm = perm[1:] + perm[:1]
            order += perm
            prev = perm[-1]
        hops = []
        on_len = int(round(duty * dwell))
        for s in range(nslot):
            a = int(round(s * dwell))
            b = min(dur, a + on_len)
            if b - a < 16:
                continue
            f = cf[order[s]]
            t = np.arange(b - a, dtype=np.float64) / rate
            seg = g[a:b] * _cis(TWO_PI * f * t + rng.uniform(0, TWO_PI))
            wave[a:b] = _ramp_edges(seg, ramp)
            bursts.append((a, b))
            hops.append(float(f))
        gen.last_hops = hops
        return wave, bursts, list(bursts)

    pr = dict(p)
    pr.update(baud=baud, chan_spacing_hz=spacing, channel_offsets_hz=[float(x) for x in cf],
              inst_bw_hz=B, span_hz=float(cf.max() - cf.min() + B))
    return dict(bw=B, symrate=baud, bands=[(float(f - B / 2), float(f + B / 2)) for f in cf], params=pr,
                min_len=int(3 * dwell), typ_len=int(40 * dwell), gen=gen)


def _k_pulsed(rate, seconds, p):
    """pw_s is the 50 %-amplitude (FWHM) pulse width; raised-cosine rise/fall (rise*pw each) extend beyond it,
    so the truth burst (base to base) lasts pw_s*(1+rise)."""
    pw, pri = float(p['pw_s']), float(p['pri_s'])
    jit = float(p['pri_jitter'])
    mod = p['mod']
    if mod not in ('none', 'lfm'):
        raise ValueError("pulsed: mod must be 'none' or 'lfm'")
    npw = int(round(pw * rate))
    if npw < 8:
        raise ValueError('pulsed: pulse is < 8 samples at this rate (pw_s*rate = %.1f)' % (pw * rate))
    if pri < 1.5 * pw * (1 + float(p['rise'])):
        raise ValueError('pulsed: pri_s must be >= 1.5*pw_s*(1+rise)')
    cbw = float(p['chirp_bw_hz']) if p['chirp_bw_hz'] else (0.2 * rate if mod == 'lfm' else 0.0)
    B = _BW_PULSE / pw + (cbw if mod == 'lfm' else 0.0)
    if B >= 0.9 * rate:
        raise ValueError('pulsed: occupied bandwidth %.0f Hz does not fit at this rate' % B)
    rise = max(2, int(round(float(p['rise']) * npw)))
    ntot = npw + rise
    r = (0.5 - 0.5 * np.cos(np.pi * (np.arange(rise) + 0.5) / rise)).astype(np.float32)
    env = np.concatenate([r, np.ones(npw - rise, np.float32), r[::-1]])
    t = (np.arange(ntot, dtype=np.float64) - rise / 2.0) / rate      # chirp spans the FWHM interval
    ph = (math.pi * (cbw / pw) * t * t - math.pi * cbw * t) if mod == 'lfm' else np.zeros(ntot)
    pulse = (env * _cis(ph)).astype(np.complex64)
    pri_n = pri * rate

    def gen(rng, dur):
        n = int((dur - ntot) // pri_n) + 1
        starts = np.round(np.arange(n) * pri_n + rng.uniform(0, pri_n * 0.5)).astype(np.int64)
        if jit > 0:
            starts = starts + np.round(rng.uniform(-jit, jit, n) * pri_n).astype(np.int64)
        starts = starts[(starts >= 0) & (starts + ntot <= dur)]
        wave = np.zeros(dur, np.complex64)
        for i in range(ntot):
            wave[starts + i] = pulse[i]
        b = [(int(s), int(s) + ntot) for s in starts]
        return wave, b, list(b)

    pr = dict(p)
    pr.update(duty=pw / pri, chirp_bw_hz=cbw, pulse_extent_s=ntot / rate)
    return dict(bw=B, symrate=None, bands=[(-B / 2, B / 2)], params=pr, min_len=int(2 * pri_n + ntot),
                typ_len=int(20 * pri_n), gen=gen)


_PREP = {'noise': _k_noise, 'cw': _k_cw, 'am': _k_am, 'nfm': _k_nfm, 'wfm': _k_wfm, 'ook': _k_ook,
         'fsk2': _k_fsk2, 'gfsk': _k_gfsk, 'bpsk': _k_psk('bpsk'), 'qpsk': _k_psk('qpsk'),
         'lora': _k_lora, 'ofdm': _k_ofdm, 'hopper': _k_hopper, 'pulsed': _k_pulsed}


def _prep(kind, rate, seconds, kw):
    if kind not in _PREP:
        raise ValueError('unknown kind %r; choose from %s' % (kind, ', '.join(KINDS)))
    d = dict(_DEF[kind])
    d['occ_bw_hz'] = None
    bad = set(kw) - set(d)
    if bad:
        raise TypeError('%s: unexpected kw %s; valid: %s' % (kind, sorted(bad), sorted(d)))
    d.update(kw)
    occ = d.pop('occ_bw_hz')
    spec = _PREP[kind](rate, seconds, d)
    spec['kind'] = kind
    spec['bw_nominal'] = spec['bw']
    if occ is not None and spec['bw'] is not None:
        spec['bw'] = float(occ)
        spec['params']['occ_bw_hz'] = float(occ)
    return spec


# ----------------------------------------------------------------------------------------------
# scheduling, mixing, quantisation
# ----------------------------------------------------------------------------------------------
def _windows(rng, n, frac, min_len, typ_len):
    """Random burst windows (start, end) covering ~frac of n samples."""
    if frac >= 0.9999 or n <= min_len:
        return [(0, n)]
    if frac <= 0:
        return []
    min_len = max(1, int(min_len))
    on_tot = min(n, max(int(round(frac * n)), min_len))
    K = int(round(on_tot / max(typ_len, min_len)))
    K = max(1, min(K, on_tot // min_len, 64))
    share = rng.dirichlet(np.full(K, 3.0))
    lens = np.maximum(min_len, np.floor(share * on_tot)).astype(np.int64)
    diff = int(on_tot - lens.sum())
    while diff != 0:
        i = int(np.argmax(lens)) if diff < 0 else int(np.argmin(lens))
        step = diff if diff > 0 else -min(-diff, int(lens[i] - min_len))
        if step == 0:
            break
        lens[i] += step
        diff -= step
    gap_tot = int(n - lens.sum())
    min_gap = max(1, min(int(0.15 * typ_len), gap_tot // (K + 1)))
    free = max(0, gap_tot - min_gap * (K - 1))
    gs = np.floor(rng.dirichlet(np.ones(K + 1)) * free).astype(np.int64)
    gs[1:K] += min_gap
    gs[K] += gap_tot - int(gs.sum())
    out, s = [], int(gs[0])
    for i in range(K):
        out.append((s, s + int(lens[i])))
        s += int(lens[i]) + int(gs[i + 1])
    return out


def _power_on(wave, on):
    tot, cnt = 0.0, 0
    for a, b in on:
        for s in range(a, b, _BLK):
            e = min(b, s + _BLK)
            x = wave[s:e]
            tot += float(np.einsum('i,i->', x.real, x.real, dtype=np.float64)
                         + np.einsum('i,i->', x.imag, x.imag, dtype=np.float64))
        cnt += b - a
    return tot / max(cnt, 1)


def _mix_add(sig, a, wave, fr, rate, phi, inplace=False):
    """sig[a:a+len(wave)] += wave * exp(j*(2*pi*fr*t + phi)) with t the global sample time.
    inplace=True: wave itself is rotated (sig is wave, a == 0)."""
    n = len(wave)
    for s in range(0, n, _BLK):
        e = min(n, s + _BLK)
        if fr == 0.0:
            rot = np.complex64(np.exp(1j * phi))
        else:
            idx = np.arange(a + s, a + e, dtype=np.float64)
            rot = _cis(TWO_PI * np.remainder(idx * (fr / rate), 1.0) + phi)
        if inplace:
            wave[s:e] *= rot
        else:
            sig[a + s:a + e] += wave[s:e] * rot


def _merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _plan(kind, rate, snr_db, offset_hz, on_fraction, kw):
    if kind != 'multi':
        return [dict(kind=kind, snr=float(snr_db), off=float(offset_hz), frac=float(on_fraction), kw=dict(kw))]
    d = dict(_DEF['multi'])
    bad = set(kw) - set(d)
    if bad:
        raise TypeError('multi: unexpected kw %s; valid: %s' % (sorted(bad), sorted(d)))
    d.update(kw)
    kinds = tuple(d['kinds'])
    if len(kinds) < 2 or any(k in ('multi', 'noise') for k in kinds) or any(k not in _PREP for k in kinds):
        raise ValueError("multi: kinds must be >= 2 signal kinds (not 'multi'/'noise')")
    nk = len(kinds)
    rel = d['rel_offsets_hz'] if d['rel_offsets_hz'] is not None else \
        [(i - (nk - 1) / 2.0) * 0.4 * rate for i in range(nk)]
    snrs = d['snrs_db'] if d['snrs_db'] is not None else [snr_db] * nk
    fr = d['on_fractions'] if d['on_fractions'] is not None else [on_fraction] * nk
    sub = d['sub_kw'] if d['sub_kw'] is not None else [{}] * nk
    if not (len(rel) == len(snrs) == len(fr) == len(sub) == nk):
        raise ValueError('multi: rel_offsets_hz/snrs_db/on_fractions/sub_kw must each have len(kinds) entries')
    return [dict(kind=kinds[i], snr=float(snrs[i]), off=float(offset_hz + rel[i]), frac=float(fr[i]),
                 kw=dict(sub[i])) for i in range(nk)]


def make(kind, rate, seconds, snr_db, seed=0, offset_hz=0.0, on_fraction=1.0, dc=0.0, cfo_hz=0.0,
         rms_lsb=20.0, return_ci8=True, **kw):
    """Generate one synthetic HackRF-style ci8 capture.  See module docstring and `describe()`."""
    rate = float(rate)
    n = int(round(rate * seconds))
    if n < 64:
        raise ValueError('capture too short')
    if kind not in KINDS:
        raise ValueError('unknown kind %r; choose from %s' % (kind, ', '.join(KINDS)))
    plan = _plan(kind, rate, snr_db, offset_hz, on_fraction, kw)
    ss = np.random.SeedSequence([int(seed), 0xC0FFEE])
    children = ss.spawn(len(plan) + 1)
    noise_seq = children[-1]

    sig = None          # allocated lazily; a single full-length window becomes `sig` without a copy
    comps = []
    for ci, c in enumerate(plan):
        spec = _prep(c['kind'], rate, seconds, c['kw'])
        ct = dict(kind=c['kind'], family=FAMILY[c['kind']], snr_db=c['snr'], offset_hz=c['off'],
                  center_hz=c['off'] + cfo_hz, occupied_bw_hz=spec['bw'], symbol_rate=spec['symrate'],
                  parameters=spec['params'], bursts=[], on_intervals=[], bands_hz=[], on_fraction=0.0)
        if spec['gen'] is None:      # noise
            comps.append(ct)
            continue
        for lo, hi in spec['bands']:
            flo, fhi = c['off'] + cfo_hz + lo, c['off'] + cfo_hz + hi
            if flo <= -rate / 2 or fhi >= rate / 2:
                raise ValueError('%s: signal band [%.0f, %.0f] Hz does not fit in +-%.0f Hz (offset/cfo too large)'
                                 % (c['kind'], flo, fhi, rate / 2))
            ct['bands_hz'].append((flo, fhi))
        rng = np.random.default_rng(children[ci])
        wins = _windows(rng, n, c['frac'], spec['min_len'], spec['typ_len'])
        amp = math.sqrt(10.0 ** (c['snr'] / 10.0) * spec['bw'] / rate)   # noise power/sample is 1 (relative)
        bursts, on_all, hops = [], [], []
        ct['windows'] = [(a / rate, b / rate) for a, b in wins]
        ct['gate_fraction'] = sum(b - a for a, b in wins) / n
        for (a, b) in wins:
            if b - a < spec['min_len']:
                raise ValueError('%s: capture too short for one burst (need >= %.4f s)'
                                 % (c['kind'], spec['min_len'] / rate))
            wave, bs, on = spec['gen'](rng, b - a)
            p_on = _power_on(wave, on)
            if p_on <= 0:
                raise RuntimeError('empty burst')
            wave *= np.float32(amp / math.sqrt(p_on))
            if sig is None and a == 0 and b == n:
                _mix_add(wave, 0, wave, c['off'] + cfo_hz, rate, rng.uniform(0, TWO_PI), inplace=True)
                sig = wave
            else:
                if sig is None:
                    sig = np.zeros(n, np.complex64)
                _mix_add(sig, a, wave, c['off'] + cfo_hz, rate, rng.uniform(0, TWO_PI))
            bursts += [(a + s, a + e) for s, e in bs]
            on_all += [(a + s, a + e) for s, e in on]
            if c['kind'] == 'hopper':
                hops += [c['off'] + cfo_hz + f for f in spec['gen'].last_hops]
        on_all = _merge(on_all)
        ct['bursts'] = [(s / rate, e / rate) for s, e in bursts]
        ct['on_intervals'] = [(s / rate, e / rate) for s, e in on_all]
        ct['on_fraction'] = sum(e - s for s, e in on_all) / n
        if hops:
            ct['hop_freqs_hz'] = hops
        comps.append(ct)

    if sig is None:
        sig = np.zeros(n, np.complex64)
    # ---- scaling to rms_lsb (two passes over deterministic per-chunk noise) -------------------
    nseeds = noise_seq.spawn((n + _BLK - 1) // _BLK)
    cache = n * 8 <= 200e6
    cached = []

    def noise_chunk(i, m):
        r = np.random.default_rng(nseeds[i])
        z = r.standard_normal((m, 2), dtype=np.float32)
        out = np.empty(m, np.complex64)
        out.real = z[:, 0]
        out.imag = z[:, 1]
        out *= np.float32(math.sqrt(0.5))
        return out

    energy = 0.0
    for i, a in enumerate(range(0, n, _BLK)):
        b = min(n, a + _BLK)
        nz = noise_chunk(i, b - a)
        if cache:
            cached.append(nz)
        y = sig[a:b] + nz
        energy += float(np.einsum('i,i->', y.real, y.real, dtype=np.float64)
                        + np.einsum('i,i->', y.imag, y.imag, dtype=np.float64))
    k = float(rms_lsb) / math.sqrt(energy / (2.0 * n))

    dcc = complex(dc, dc) if np.isscalar(dc) and not isinstance(dc, complex) else complex(dc)
    iq = np.empty(n, np.complex64)
    q = np.empty((n, 2), np.int8)
    nclip = 0
    for i, a in enumerate(range(0, n, _BLK)):
        b = min(n, a + _BLK)
        nz = cached[i] if cache else noise_chunk(i, b - a)
        y = (sig[a:b] + nz) * np.float32(k)
        re = np.rint(y.real + np.float32(dcc.real))
        im = np.rint(y.imag + np.float32(dcc.imag))
        nclip += int(np.count_nonzero((re > 127) | (re < -128) | (im > 127) | (im < -128)))
        np.clip(re, -128, 127, out=re)
        np.clip(im, -128, 127, out=im)
        q[a:b, 0] = re
        q[a:b, 1] = im
        iq[a:b].real = re
        iq[a:b].imag = im
    ci8 = q.tobytes() if return_ci8 else None

    # ---- truth --------------------------------------------------------------------------------
    clip_fraction = nclip / n
    warnings = []
    if clip_fraction > 1e-3:
        warnings.append('clip_fraction %.4f > 0.1%%: lower snr_db/duty or raise rms headroom' % clip_fraction)
    for ct in comps:
        sn = 10.0 ** (ct['snr_db'] / 10.0) * (ct['occupied_bw_hz'] or 0.0) / rate
        ct['on_power_lsb2'] = float(k * k * sn) if ct['occupied_bw_hz'] else 0.0
    truth = dict(
        kind=kind, family=FAMILY[kind], rate=rate, seconds=seconds, n=n, seed=int(seed),
        snr_db=float(snr_db), offset_hz=float(offset_hz), cfo_hz=float(cfo_hz),
        center_hz=float(offset_hz + cfo_hz), dc_lsb=complex(dcc), rms_lsb=float(rms_lsb),
        noise_rms_lsb=float(k * math.sqrt(0.5)), noise_psd_lsb2_per_hz=float(k * k / rate),
        clip_fraction=clip_fraction, warnings=warnings,
        snr_definition='in-band, on-state: P_on / (N0 * occupied_bw_hz)',
    )
    if len(comps) == 1:
        truth.update(comps[0])
        truth['kind'], truth['family'] = kind, FAMILY[kind]
        truth['snr_db'], truth['offset_hz'] = float(snr_db), float(offset_hz)
        truth['center_hz'] = float(offset_hz + cfo_hz)
    else:
        allb = [b for c in comps for b in c['bands_hz']]
        truth.update(
            occupied_bw_hz=max(b[1] for b in allb) - min(b[0] for b in allb),
            symbol_rate=None, parameters=dict(kind_list=[c['kind'] for c in comps]),
            bursts=_merge([iv for c in comps for iv in c['bursts']]),
            on_intervals=_merge([iv for c in comps for iv in c['on_intervals']]),
            bands_hz=allb, on_fraction=float(np.mean([c['on_fraction'] for c in comps])),
            windows=_merge([iv for c in comps for iv in c.get('windows', [])]),
            gate_fraction=float(np.mean([c.get('gate_fraction', 0.0) for c in comps])),
            components=comps)
    return dict(iq=iq, ci8=ci8, truth=truth)


def on_mask(truth, key='on_intervals'):
    """Boolean per-sample mask of truth[key] ('on_intervals', 'bursts' or 'windows')."""
    m = np.zeros(truth['n'], bool)
    for a, b in truth[key]:
        m[int(round(a * truth['rate'])):int(round(b * truth['rate']))] = True
    return m


def write_sigmf(res, base_path, center_hz=433.92e6):
    """Write `base_path.sigmf-data` (raw ci8) and a minimal `.sigmf-meta` (same core: fields hackrf-scout
    writes; the truth is stored under `iqchar:truth`).  Returns the two paths."""
    import json
    t = res['truth']
    data, meta = base_path + '.sigmf-data', base_path + '.sigmf-meta'
    with open(data, 'wb') as f:
        f.write(res['ci8'])
    doc = {'global': {'core:datatype': 'ci8', 'core:sample_rate': float(t['rate']), 'core:version': '1.2.2',
                      'core:recorder': 'iqchar.gen', 'core:hw': 'synthetic',
                      'iqchar:truth': json.loads(json.dumps(t, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else float(o)))},
           'captures': [{'core:sample_start': 0, 'core:frequency': float(center_hz)}], 'annotations': []}
    with open(meta, 'w') as f:
        json.dump(doc, f)
    return data, meta


def write_ci8(res, path):
    """Write res['ci8'] to `path` (raw hackrf_transfer-style file)."""
    with open(path, 'wb') as f:
        f.write(res['ci8'])


def describe():
    """Human readable API summary (kinds, kw names and defaults)."""
    lines = ["make(kind, rate, seconds, snr_db, seed=0, offset_hz=0.0, on_fraction=1.0, dc=0.0, cfo_hz=0.0,",
             "     rms_lsb=20.0, return_ci8=True, **kw) -> {'iq','ci8','truth'}",
             "every kind also accepts occ_bw_hz=None (override B in the SNR definition)"]
    for k in KINDS:
        lines.append('  %-7s %s' % (k, _DEF.get(k, {})))
    return '\n'.join(lines)


if __name__ == '__main__':
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import selftest
    sys.exit(selftest.main(sys.argv[1:]))

#!/usr/bin/env python3
"""Self-test of gen.py.  Run:  python3 gen.py        (or python3 selftest.py [--quick] [--long] [--json out.json])

Per kind, 3 seeds (different SNR / offset), rate 2 Msps (OFDM, GFSK, hopper, pulsed also at 10 Msps):
  * independent Welch spectrum estimate -> in-band on-state SNR within 1.5 dB of the request
  * spectral centroid at offset_hz (+cfo) within max(8 % of span, 4 bins)
  * clipping <= 0.1 % of samples (recomputed from the returned samples), RMS == rms_lsb (+-1.5 %)
  * ci8 round trip is bit exact and has 2*n bytes
  * truth on-intervals vs a channel-filter + smoothed-envelope + Otsu threshold detector
  * truth bookkeeping (noise PSD, on-state power) vs measurement; nominal occupied bandwidth contains >= 97 % of power
plus: determinism, DC, CFO, on_fraction gating, error handling, timing.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402
import meas  # noqa: E402

SNR_TOL_DB = 1.5
SEED_SNRS = (6.0, 12.0, 17.0)
SEED_OFFS = (-0.8, 0.1, 0.6)       # x max_offset, where max_offset = half the free margin around the band

# name, kind, kw, [(rate, seconds)], per-case SNR list override, containment-test SNR
MATRIX = [
    ('cw', 'cw', {}, [(2e6, 1.0)], None, 40),
    ('am-tone', 'am', {}, [(2e6, 1.0)], None, 40),
    ('am-voice', 'am', {'audio': 'voice'}, [(2e6, 1.0)], None, 40),
    ('nfm-voice', 'nfm', {}, [(2e6, 1.0)], None, 40),
    ('nfm-tone', 'nfm', {'audio': 'tone'}, [(2e6, 1.0)], None, 40),
    ('wfm', 'wfm', {}, [(2e6, 1.0)], None, 40),
    ('ook-pwm', 'ook', {}, [(2e6, 1.0)], None, 40),
    ('ook-manchester', 'ook', {'coding': 'manchester', 'baud': 4000.0}, [(2e6, 1.0)], None, 40),
    ('fsk2-rect', 'fsk2', {'shape': 'rect'}, [(2e6, 1.0)], None, 40),
    ('fsk2-gauss', 'fsk2', {'shape': 'gauss'}, [(2e6, 1.0)], None, 40),
    ('gfsk', 'gfsk', {}, [(10e6, 0.3), (4e6, 0.3)], None, 40),
    ('bpsk', 'bpsk', {}, [(2e6, 1.0)], None, 40),
    ('qpsk', 'qpsk', {}, [(2e6, 1.0)], None, 40),
    ('lora-sf7', 'lora', {'sf': 7}, [(2e6, 1.0)], None, 40),
    ('lora-sf9', 'lora', {'sf': 9}, [(2e6, 1.2)], None, 40),
    ('ofdm', 'ofdm', {}, [(2e6, 0.5), (10e6, 0.25)], None, 30),
    ('ofdm-narrow', 'ofdm', {'bw_hz': 600e3, 'mod': '16qam'}, [(2e6, 0.5)], None, 30),
    ('hopper', 'hopper', {}, [(2e6, 0.5), (10e6, 0.25)], None, 40),
    ('pulsed', 'pulsed', {}, [(2e6, 1.0)], (6.0, 10.0, 12.0), 30),
    ('pulsed-fast', 'pulsed', {'pw_s': 2e-6, 'pri_s': 100e-6}, [(10e6, 0.3)], (6.0, 9.0, 11.0), 30),
    ('pulsed-lfm', 'pulsed', {'mod': 'lfm', 'pw_s': 50e-6, 'pri_s': 2e-3}, [(2e6, 1.0)], (6.0, 10.0, 12.0), 30),
    ('multi-cw+nfm', 'multi', {'kinds': ('cw', 'nfm'), 'rel_offsets_hz': (-350e3, 250e3)}, [(2e6, 1.0)], None, 40),
    ('multi-ook+qpsk', 'multi', {'kinds': ('ook', 'qpsk'), 'rel_offsets_hz': (-400e3, 300e3),
                                  'snrs_db': (14.0, 12.0)}, [(2e6, 1.0)], None, 40),
]


class Report:
    def __init__(self):
        self.rows = []
        self.fail = []
        self.worst = {}

    def check(self, ok, msg):
        if not ok:
            self.fail.append(msg)
            print('   FAIL:', msg)
        return ok

    def note(self, key, val, better='max'):
        cur = self.worst.get(key)
        if cur is None or (val > cur if better == 'max' else val < cur):
            self.worst[key] = val


def free_margin(kind, rate, kw):
    plan = gen._plan(kind, rate, 10.0, 0.0, 1.0, dict(kw))
    half = 0.0
    for c in plan:
        spec = gen._prep(c['kind'], rate, 1.0, dict(c['kw']))
        for lo, hi in spec['bands']:
            half = max(half, abs(c['off'] + lo), abs(c['off'] + hi))
    return 0.5 * (rate / 2 - half)


def sample_checks(R, res, tag, rep, kw_rms=20.0, dc=0.0):
    t = res['truth']
    iq, ci8 = res['iq'], res['ci8']
    n = t['n']
    rep.check(iq.dtype == np.complex64 and len(iq) == n, '%s: iq dtype/len' % tag)
    rep.check(isinstance(ci8, (bytes, bytearray)) and len(ci8) == 2 * n, '%s: ci8 length' % tag)
    raw = np.frombuffer(ci8, np.int8).reshape(-1, 2)
    rt = raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)
    rep.check(np.array_equal(rt, iq), '%s: ci8 does not round-trip to iq' % tag)
    rails = float(np.mean((raw[:, 0] >= 127) | (raw[:, 0] <= -128) | (raw[:, 1] >= 127) | (raw[:, 1] <= -128)))
    rep.note('clip_fraction', rails)
    rep.check(rails <= 1e-3, '%s: clip %.5f > 0.1%%' % (tag, rails))
    rms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))
    rep.check(abs(rms / kw_rms - 1) < 0.015 or dc != 0, '%s: rms %.3f vs %.1f' % (tag, rms, kw_rms))
    rep.note('rms_err', abs(rms / kw_rms - 1))
    return rms


def analyse(res, tag, rep, snr_floor_for_env=10.0, enforce_env=True):
    """SNR / centroid / bookkeeping / envelope checks for one result (single or multi)."""
    t = res['truth']
    x = res['iq']
    comps = t.get('components', [t])
    allb = t['bands_hz']
    out = []
    for ci, c in enumerate(comps):
        if c['kind'] == 'noise':
            continue
        e = meas.snr_estimate(x, t, bands=c['bands_hz'], bw=c['occupied_bw_hz'],
                              duty=c['on_fraction'], all_bands=allb)
        err = e['snr_db'] - c['snr_db']
        span = max(b[1] for b in c['bands_hz']) - min(b[0] for b in c['bands_hz'])
        cen_err = e['centroid_hz'] - c['center_hz']
        cen_tol = max(0.08 * span, 4 * e['df'])
        # bookkeeping: noise PSD (truth) vs measured (measured includes ~1/6 LSB^2 quantisation noise)
        n0_t = t['noise_psd_lsb2_per_hz'] + (1.0 / 6.0) / t['rate']
        n0_err_db = 10 * np.log10(e['n0'] / n0_t)
        pon_meas = e['p_sig'] / e['duty']
        pon_err_db = 10 * np.log10(pon_meas / c['on_power_lsb2'])
        rep.note('snr_err_db', abs(err))
        rep.note('centroid_err_over_tol', abs(cen_err) / cen_tol)
        rep.note('n0_err_db', abs(n0_err_db))
        rep.note('pon_err_db', abs(pon_err_db))
        rep.check(abs(err) <= SNR_TOL_DB, '%s[%s]: SNR est %.2f dB vs request %.2f (err %+.2f)'
                  % (tag, c['kind'], e['snr_db'], c['snr_db'], err))
        rep.check(abs(cen_err) <= cen_tol, '%s[%s]: centroid error %.0f Hz (tol %.0f)' % (tag, c['kind'], cen_err, cen_tol))
        rep.check(abs(n0_err_db) <= 0.5, '%s: truth noise PSD vs measured %.2f dB' % (tag, n0_err_db))
        rep.check(abs(pon_err_db) <= 1.5, '%s[%s]: truth on-power vs measured %.2f dB' % (tag, c['kind'], pon_err_db))
        row = dict(kind=c['kind'], snr_req=c['snr_db'], snr_est=e['snr_db'], err=err, cen_err_hz=cen_err,
                   duty=c['on_fraction'], n0_err_db=n0_err_db, pon_err_db=pon_err_db)
        # envelope / burst agreement
        multi_on = len(c['on_intervals']) > 1 or c['on_fraction'] < 0.98
        if multi_on:
            ev = meas.envelope_events(x, t, bands=c['bands_hz'], on_intervals=c['on_intervals'], n0=e['n0'], p_on=e['p_sig'] / e['duty'])
            if ev is not None:
                row['env'] = ev
                if enforce_env and c['snr_db'] >= snr_floor_for_env:
                    rep.note('env_recall', ev['recall'], 'min')
                    rep.note('env_precision', ev['precision'], 'min')
                    rep.check(ev['recall'] >= 0.9, '%s[%s]: envelope recall %.2f (%d/%d truth runs)'
                              % (tag, c['kind'], ev['recall'], ev['n_truth'], ev['n_det']))
                    rep.check(ev['precision'] >= 0.9, '%s[%s]: envelope precision %.2f' % (tag, c['kind'], ev['precision']))
                    tol = max(2 * ev['smooth_s'] + 4 * ev['res_s'], 0.4 * ev['feat_s'])
                    if ev['edge_err_s'] == ev['edge_err_s']:
                        rep.note('env_edge_err_over_tol', ev['edge_err_s'] / tol)
                        rep.check(ev['edge_err_s'] <= tol, '%s[%s]: envelope edge error %.1f us (tol %.1f)'
                                  % (tag, c['kind'], 1e6 * ev['edge_err_s'], 1e6 * tol))
        out.append(row)
    return out


def run_matrix(rep, quick):
    results = []
    for name, kind, kw, rates, snrs, csnr in MATRIX:
        for (rate, sec) in rates:
            if quick and rate != rates[0][0]:
                continue
            kwr = dict(kw)
            if kind == 'multi' and 'rel_offsets_hz' in kwr:
                pass
            try:
                mo = free_margin(kind, rate, kwr)
            except Exception as ex:  # unsupported at this rate
                print('skip %s @ %.0f: %s' % (name, rate / 1e6, ex))
                continue
            if kind == 'gfsk' and rate < 4e6:
                continue
            snr_list = snrs or SEED_SNRS
            line = []
            for seed in range(3):
                snr = snr_list[seed]
                off = SEED_OFFS[seed] * mo
                tag = '%s@%gM s%d' % (name, rate / 1e6, seed)
                t0 = time.time()
                res = gen.make(kind, rate, sec, snr, seed=seed, offset_hz=off, **kwr)
                dt = time.time() - t0
                sample_checks(rate, res, tag, rep)
                rows = analyse(res, tag, rep, enforce_env=True)
                for r in rows:
                    line.append('%+.2f' % r['err'])
                    results.append(dict(case=name, rate=rate, seed=seed, **{k: v for k, v in r.items()}))
                if res['truth']['warnings']:
                    rep.check(False, '%s: %s' % (tag, res['truth']['warnings']))
                if seed == 0:
                    # determinism
                    res2 = gen.make(kind, rate, sec, snr, seed=seed, offset_hz=off, **kwr)
                    rep.check(hashlib.md5(res2['ci8']).hexdigest() == hashlib.md5(res['ci8']).hexdigest(),
                              '%s: not deterministic' % tag)
                    res3 = gen.make(kind, rate, sec, snr, seed=seed + 100, offset_hz=off, **kwr)
                    rep.check(hashlib.md5(res3['ci8']).hexdigest() != hashlib.md5(res['ci8']).hexdigest(),
                              '%s: different seeds give the same bytes' % tag)
            # containment of the nominal bandwidth (clean, high SNR)
            if kind != 'multi':
                rc = gen.make(kind, rate, sec, csnr, seed=7, rms_lsb=(6.0 if kind == 'pulsed' else 30.0), offset_hz=0.0, **kwr)
                co = meas.containment(rc['iq'], rc['truth'])
                tcl = rc['truth']['clip_fraction']
                rep.note('containment_min', co['frac_in'], 'min')
                rep.check(co['frac_in'] >= 0.97, '%s@%gM: nominal B holds only %.4f of the power' % (name, rate / 1e6, co['frac_in']))
                cont = '%.4f (B=%.0f Hz, bw99=%.0f, bw99.9=%.0f)' % (co['frac_in'], rc['truth']['occupied_bw_hz'], co['bw99'], co['bw999'])
            else:
                cont = '-'
            print('%-16s @%4.0fM  %.2fs  snr err[dB] %s | containment %s' % (name, rate / 1e6, dt, ' '.join(line), cont))
    return results


def run_noise(rep):
    res = gen.make('noise', 2e6, 0.5, 0.0, seed=1)
    sample_checks(2e6, res, 'noise', rep)
    f, P, _ = meas.welch(res['iq'], 2e6, 1024)
    flat = 10 * np.log10(P.max() / P.min())
    ps = np.convolve(P, np.ones(16) / 16, 'valid')
    flat_s = 10 * np.log10(ps.max() / ps.min())
    mean = abs(res['iq'].mean())
    rep.check(flat_s < 1.0, 'noise: smoothed PSD ripple %.2f dB' % flat_s)
    rep.check(mean < 0.2, 'noise: mean %.3f' % mean)
    rep.check(res['truth']['bursts'] == [] and res['truth']['occupied_bw_hz'] is None, 'noise: truth')
    print('noise            rms ok, PSD ripple (16-bin smoothed) %.2f dB, |mean| %.3f LSB' % (flat_s, mean))


def run_special(rep):
    # DC
    r = gen.make('cw', 2e6, 0.5, 12.0, seed=1, offset_hz=300e3, dc=6.0)
    m = r['iq'].mean()
    rep.check(abs(m - (6 + 6j)) < 0.3, 'dc scalar: mean %s' % m)
    r2 = gen.make('cw', 2e6, 0.5, 12.0, seed=1, offset_hz=300e3, dc=complex(3.0, -4.0))
    m2 = r2['iq'].mean()
    rep.check(abs(m2 - (3 - 4j)) < 0.3, 'dc complex: mean %s' % m2)
    e = meas.snr_estimate(r['iq'], r['truth'], remove_dc=True)
    rep.check(abs(e['snr_db'] - 12.0) < SNR_TOL_DB, 'dc: SNR after DC removal %.2f' % e['snr_db'])
    print('dc               mean(iq) = %.3f%+.3fj for dc=6 ; %.3f%+.3fj for dc=3-4j ; SNR after DC removal %.2f dB'
          % (m.real, m.imag, m2.real, m2.imag, e['snr_db']))
    # CFO
    r = gen.make('nfm', 2e6, 1.0, 15.0, seed=2, offset_hz=100e3, cfo_hz=37e3)
    e = meas.snr_estimate(r['iq'], r['truth'])
    rep.check(abs(e['centroid_hz'] - 137e3) < 1.5e3 and abs(r['truth']['center_hz'] - 137e3) < 1e-6,
              'cfo: centroid %.0f Hz' % e['centroid_hz'])
    print('cfo              offset 100 kHz + cfo 37 kHz -> centroid %.1f Hz (truth centre %.0f)' % (e['centroid_hz'], r['truth']['center_hz']))
    # CW peak bin position
    r = gen.make('cw', 2e6, 1.0, 20.0, seed=3, offset_hz=-412345.0)
    f, P, _ = meas.welch(r['iq'], 2e6, 1 << 16)
    pk = f[np.argmax(P)]
    rep.check(abs(pk + 412345.0) < 2 * (f[1] - f[0]), 'cw peak at %.1f' % pk)
    print('cw               peak bin %.1f Hz for offset -412345 Hz (bin %.1f Hz)' % (pk, f[1] - f[0]))
    # on_fraction gating for every kind
    worst_gate = 0.0
    for name, kind, kw, rates, snrs, csnr in MATRIX:
        rate, sec = rates[0]
        if kind == 'multi':
            continue
        kwr = dict(kw)
        mo = free_margin(kind, rate, kwr)
        snr = 13.0 if kind not in ('pulsed',) else 10.0
        sec = max(sec, 1.0 if kind in ('lora',) else sec)
        a = gen.make(kind, rate, sec, snr, seed=11, offset_hz=0.2 * mo, on_fraction=0.35, **kwr)
        b = gen.make(kind, rate, sec, snr, seed=12, offset_hz=0.2 * mo, on_fraction=0.35, **kwr)
        t = a['truth']
        tag = '%s on_fraction=0.35' % name
        sample_checks(rate, a, tag, rep)
        gf = t['gate_fraction']
        worst_gate = max(worst_gate, abs(gf - 0.35))
        rep.check(abs(gf - 0.35) <= 0.02, '%s: gate fraction %.3f' % (tag, gf))
        # bursts inside windows
        wins = t['windows']
        inside = all(any(w0 - 1e-9 <= s and e <= w1 + 1e-9 for w0, w1 in wins) for s, e in t['bursts'])
        rep.check(inside, '%s: truth burst outside its window' % tag)
        rep.check(t['windows'] != b['truth']['windows'], '%s: windows identical for different seeds' % tag)
        rep.check(t['on_fraction'] <= gf + 1e-9, '%s: on_fraction > gate' % tag)
        rows = analyse(a, tag, rep, enforce_env=True)
        r0 = rows[0]
        extra = ''
        if 'env' in r0:
            ev = r0['env']
            extra = ' env recall %.2f prec %.2f edge %.0fus (%d runs)' % (ev['recall'], ev['precision'], 1e6 * ev['edge_err_s'], ev['n_truth'])
        print('gated %-16s gate=%.3f on=%.3f nwin=%d snr err %+.2f dB%s' % (name, gf, t['on_fraction'], len(wins), r0['err'], extra))
    # errors
    for fn, exc in [(lambda: gen.make('gfsk', 2e6, 0.1, 10), ValueError),
                    (lambda: gen.make('cw', 2e6, 0.1, 10, bogus=1), TypeError),
                    (lambda: gen.make('cw', 2e6, 0.1, 10, offset_hz=1.2e6), ValueError),
                    (lambda: gen.make('nope', 2e6, 0.1, 10), ValueError),
                    (lambda: gen.make('ofdm', 2e6, 0.1, 10, bw_hz=1.7e6), ValueError)]:
        try:
            fn()
            rep.check(False, 'expected %s' % exc.__name__)
        except exc:
            pass
    r = gen.make('cw', 2e6, 0.1, 10, return_ci8=False)
    rep.check(r['ci8'] is None, 'return_ci8=False')
    print('errors           gfsk@2M, unknown kw, band beyond Nyquist, unknown kind, ofdm bw>0.8*rate all rejected')
    # explicit SNR definition test: with the same snr_db, doubling B must double the measured signal-to-N0 ratio
    ra = gen.make('nfm', 2e6, 1.0, 10.0, seed=4, occ_bw_hz=12.5e3)
    rb = gen.make('nfm', 2e6, 1.0, 10.0, seed=4, occ_bw_hz=25e3)
    ea = meas.snr_estimate(ra['iq'], ra['truth'])
    eb = meas.snr_estimate(rb['iq'], rb['truth'])
    ratio = 10 * np.log10((eb['p_sig'] / eb['n0']) / (ea['p_sig'] / ea['n0']))
    rep.check(abs(ratio - 10 * np.log10(2.0)) < 0.3 and abs(ea['snr_db'] - 10) < 1 and abs(eb['snr_db'] - 10) < 1,
              'occ_bw_hz override: Psig/N0 ratio %.2f dB' % ratio)
    print('snr-def          same snr_db, B 12.5 kHz -> 25 kHz: measured Psig/N0 rises %.2f dB (expect 3.01); est %.2f / %.2f dB'
          % (ratio, ea['snr_db'], eb['snr_db']))
    # same snr_db at different rates -> same in-band SNR (rate independence)
    ests = []
    for rate in (2e6, 10e6):
        r = gen.make('nfm', rate, 0.4, 12.0, seed=5, offset_hz=50e3)
        ests.append(meas.snr_estimate(r['iq'], r['truth'])['snr_db'])
    rep.check(max(ests) - min(ests) < 1.0, 'rate independence %s' % ests)
    print('snr-def          nfm snr_db=12 measured %.2f dB @2 Msps, %.2f dB @10 Msps' % tuple(ests))



# ---------------------------------------------------------------------------------------------
# truth consistency: do the labelled parameters (baud, deviation, depth, hop freq, ...) appear in the samples?
# ---------------------------------------------------------------------------------------------
def _chan(x, rate, fc, cutoff):
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    y = x * np.exp(-2j * np.pi * np.remainder(t * (fc / rate), 1.0)).astype(np.complex64)
    Y = np.fft.fft(y)
    f = np.fft.fftfreq(n, 1.0 / rate)
    H = 0.5 - 0.5 * np.cos(np.pi * np.clip((1.3 * cutoff - np.abs(f)) / (0.3 * cutoff), 0, 1))
    return np.fft.ifft(Y * H.astype(np.float32))


def _finst(y, rate):
    return np.angle(y[1:] * np.conj(y[:-1])) * rate / (2 * np.pi)


def _run_lengths(mask):
    st, en = meas._runs(mask)
    return (en - st)


def _quarter_cluster(lens):
    lens = np.asarray(lens, float)
    lens = lens[lens >= 0.25 * np.median(lens)]
    q = np.percentile(lens, 25)
    sel = lens[lens < 1.5 * q]
    return float(np.median(sel))


def run_consistency(rep):
    out = {}

    def rel(a, b):
        return abs(a / b - 1.0)

    # OOK chip rate from keyed run lengths
    for coding, baud in (('pwm', 3000.0), ('manchester', 2400.0)):
        r = gen.make('ook', 2e6, 0.4, 28.0, seed=3, coding=coding, baud=baud)
        t = r['truth']
        y = _chan(r['iq'], 2e6, t['center_hz'], 3 * baud)
        e = np.abs(y)
        on = e > 0.5 * np.percentile(e, 99.5)
        lens = _run_lengths(on)
        chip = _quarter_cluster(lens)
        est = 2e6 / chip
        rep.check(rel(est, t['symbol_rate']) < 0.03, 'ook %s: chip rate %.1f vs truth %.1f' % (coding, est, t['symbol_rate']))
        out['ook-' + coding] = (est, t['symbol_rate'])
        print('consistency      ook-%-10s chip rate measured %.1f Hz, truth %.1f Hz' % (coding, est, t['symbol_rate']))
    # FSK / GFSK baud and deviation
    for kind, rate, kw, dev in (('fsk2', 2e6, {'shape': 'rect'}, 25e3), ('fsk2', 2e6, {'shape': 'gauss'}, 25e3),
                                ('gfsk', 10e6, {}, 250e3)):
        r = gen.make(kind, rate, 0.3, 30.0, seed=4, **kw)
        t = r['truth']
        cut = 0.55 * t['occupied_bw_hz']
        y = _chan(r['iq'], rate, t['center_hz'], cut)
        f = _finst(y, rate)
        if kind == 'gfsk':        # only inside packets
            m = np.zeros(len(f), bool)
            for a, b in t['on_intervals']:
                m[int(a * rate) + 20:int(b * rate) - 20] = True
        else:
            m = np.ones(len(f), bool)
        ff = np.convolve(f, np.ones(5) / 5, 'same') if kind != 'gfsk' else np.convolve(f, np.ones(3) / 3, 'same')
        lens = _run_lengths((ff > 0) & m)
        lens2 = _run_lengths((ff <= 0) & m)
        est = rate / _quarter_cluster(np.concatenate([lens, lens2]))
        dev_est = float(np.median(np.abs(ff[m]))) if kw.get('shape') == 'rect' else float(np.percentile(np.abs(ff[m]), 90))
        rep.check(rel(est, t['symbol_rate']) < 0.03, '%s %s: baud %.1f vs %.1f' % (kind, kw, est, t['symbol_rate']))
        rep.check(rel(dev_est, dev) < (0.06 if kind == 'fsk2' else 0.15), '%s %s: dev %.0f vs %.0f' % (kind, kw, dev_est, dev))
        print('consistency      %-5s %-18s baud measured %.1f vs truth %.1f ; |dev| %.0f Hz (nominal %.0f)'
              % (kind, kw, est, t['symbol_rate'], dev_est, dev))
    # PSK symbol-rate cyclostationary line
    for kind in ('bpsk', 'qpsk'):
        r = gen.make(kind, 2e6, 1.0, 25.0, seed=5, symrate=250e3)
        t = r['truth']
        y = _chan(r['iq'], 2e6, t['center_hz'], 0.6 * t['occupied_bw_hz'])
        e = np.abs(y) ** 2
        E = np.abs(np.fft.rfft(e - e.mean()))
        fr = np.fft.rfftfreq(len(e), 1 / 2e6)
        band = (fr > 0.8 * t['symbol_rate']) & (fr < 1.2 * t['symbol_rate'])
        pk = fr[band][np.argmax(E[band])]
        rep.check(abs(pk - t['symbol_rate']) < 20.0, '%s: symbol-rate line %.1f vs %.1f' % (kind, pk, t['symbol_rate']))
        print('consistency      %-5s symbol-rate line in |x|^2 at %.1f Hz (truth %.1f)' % (kind, pk, t['symbol_rate']))
    # LoRa chirp period via autocorrelation on the preamble
    for sf in (7, 9):
        r = gen.make('lora', 2e6, 0.6, 25.0, seed=6, sf=sf)
        t = r['truth']
        T = (1 << sf) / 125e3
        a, b = t['bursts'][0]
        seg = r['iq'][int(a * 2e6):int((a + 5 * T) * 2e6)]
        y = _chan(seg, 2e6, t['center_hz'], 70e3)
        Ls = np.arange(int(0.9 * T * 2e6), int(1.1 * T * 2e6))
        c = np.array([abs(np.vdot(y[:len(y) - L], y[L:])) for L in Ls[::4]])
        # refine around the best coarse lag
        L0 = Ls[::4][int(np.argmax(c))]
        Lf = np.arange(L0 - 4, L0 + 5)
        cf = np.array([abs(np.vdot(y[:len(y) - L], y[L:])) for L in Lf])
        Lbest = Lf[int(np.argmax(cf))]
        rate_est = 125e3 / ((Lbest / 2e6) * 125e3) / (1 << sf) * (1 << sf) / (1 << sf)
        sym_est = 1.0 / (Lbest / 2e6)
        rep.check(abs(sym_est / t['symbol_rate'] - 1) < 0.003, 'lora sf%d: symbol rate %.2f vs %.2f' % (sf, sym_est, t['symbol_rate']))
        print('consistency      lora sf%d preamble chirp autocorrelation -> symbol rate %.2f Hz (truth %.2f)' % (sf, sym_est, t['symbol_rate']))
    # OFDM symbol period from cyclic-prefix correlation
    for rate in (2e6, 10e6):
        r = gen.make('ofdm', rate, 0.1, 25.0, seed=8)
        t = r['truth']
        N = t['parameters']['n_fft']
        L = N + t['parameters']['cp_samples']
        best = max(t['bursts'], key=lambda ab: ab[1] - ab[0])
        seg = r['iq'][int(best[0] * rate):int(best[1] * rate)]
        seg = seg[int(5.5 * N):]                       # skip preamble
        prod = seg[N:] * np.conj(seg[:-N])
        c = np.abs(np.convolve(prod, np.ones(t['parameters']['cp_samples']), 'valid'))   # coherent sum over one CP
        c = c - c.mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(c, 2 * len(c))) ** 2)[:len(c)]
        lo, hi = int(0.6 * L), int(1.4 * L)
        lag = lo + int(np.argmax(ac[lo:hi]))
        est = rate / lag
        rep.check(abs(lag - L) <= 1, 'ofdm @%gM: CP symbol period %d vs %d samples' % (rate / 1e6, lag, L))
        print('consistency      ofdm @%2.0fM  symbol period from CP correlation %d samples (truth %d), rate %.1f Hz (truth %.1f)'
              % (rate / 1e6, lag, L, est, t['symbol_rate']))
    # hopper per-hop frequency
    for rate in (2e6, 10e6):
        r = gen.make('hopper', rate, 0.1, 22.0, seed=9)
        t = r['truth']
        errs = []
        for (a, b), fh in list(zip(t['bursts'], t['hop_freqs_hz']))[:40]:
            seg = r['iq'][int(a * rate):int(b * rate)]
            nf = 1 << int(np.ceil(np.log2(len(seg))))
            P = np.abs(np.fft.fft(seg * np.hanning(len(seg)), nf)) ** 2
            f = np.fft.fftfreq(nf, 1 / rate)
            P = np.fft.fftshift(P)
            f = np.fft.fftshift(f)
            wgt = np.maximum(P - np.median(P) * 1.5, 0)
            errs.append(abs((f * wgt).sum() / wgt.sum() - fh))
        sp = t['parameters']['chan_spacing_hz']
        rep.check(max(errs) < 0.12 * sp, 'hopper @%gM: hop frequency error %.0f Hz (spacing %.0f)' % (rate / 1e6, max(errs), sp))
        dw = np.median(np.diff([a for a, b in t['bursts']]))
        rep.check(abs(dw - 625e-6) < 3e-6 or abs(dw - 2 * 625e-6) < 3e-6, 'hopper dwell %.1f us' % (dw * 1e6))
        print('consistency      hopper @%2.0fM  worst per-hop centroid error %.0f Hz over 40 hops (spacing %.0f Hz); burst period %.1f us'
              % (rate / 1e6, max(errs), sp, dw * 1e6))
    # AM depth and tone
    r = gen.make('am', 2e6, 0.5, 30.0, seed=10)
    t = r['truth']
    y = _chan(r['iq'], 2e6, t['center_hz'], 1.2e3)
    e = np.abs(y)
    hi, lo = np.percentile(e, 99.0), np.percentile(e, 1.0)
    depth = (hi - lo) / (hi + lo)
    E = np.abs(np.fft.rfft(e - e.mean()))
    fr = np.fft.rfftfreq(len(e), 1 / 2e6)
    ftone = fr[1:][np.argmax(E[1:])]
    rep.check(abs(depth - 0.8) < 0.04 and abs(ftone - 1000) < 5, 'am depth %.3f tone %.1f' % (depth, ftone))
    print('consistency      am   modulation depth %.3f (0.80), tone %.1f Hz (1000)' % (depth, ftone))
    # NFM tone deviation
    r = gen.make('nfm', 2e6, 0.5, 40.0, seed=11, audio='tone')
    t = r['truth']
    y = _chan(r['iq'], 2e6, t['center_hz'], 5e3)
    f = np.convolve(_finst(y, 2e6), np.ones(41) / 41, 'same')
    devm = (np.percentile(f, 99.5) - np.percentile(f, 0.5)) / 2
    rep.check(abs(devm / 2500 - 1) < 0.06, 'nfm dev %.0f' % devm)
    print('consistency      nfm  tone peak deviation %.0f Hz (2500)' % devm)
    # WFM pilot
    r = gen.make('wfm', 2e6, 0.5, 30.0, seed=12)
    t = r['truth']
    y = _chan(r['iq'], 2e6, t['center_hz'], 120e3)[::4]
    f = _finst(y, 2e6 / 4)
    F = np.abs(np.fft.rfft(f * np.hanning(len(f)))) * 2 / np.sum(np.hanning(len(f)))
    fr = np.fft.rfftfreq(len(f), 4 / 2e6)
    band = (fr > 18e3) & (fr < 20e3)
    pk = fr[band][np.argmax(F[band])]
    amp = F[band].max()
    rep.check(abs(pk - 19e3) < 20 and abs(amp / 6750 - 1) < 0.2, 'wfm pilot %.0f Hz amp %.0f' % (pk, amp))
    print('consistency      wfm  pilot at %.1f Hz with %.0f Hz deviation (nominal 19000 / 6750)' % (pk, amp))
    # pulsed pw / PRI
    for kw, rate in (({}, 2e6), ({'pw_s': 2e-6, 'pri_s': 100e-6}, 10e6)):
        r = gen.make('pulsed', rate, 0.3, 24.0, seed=13, **kw)
        t = r['truth']
        y = _chan(r['iq'], rate, t['center_hz'], 0.55 * t['occupied_bw_hz'])
        e = np.abs(y) ** 2
        on = e > 0.25 * np.percentile(e, 99.9)
        st, en = meas._runs(on)
        pw = float(np.median(en - st)) / rate
        pri = float(np.median(np.diff(st))) / rate
        pwt = t['parameters']['pw_s']
        prit = t['parameters']['pri_s']
        rep.check(abs(pw / pwt - 1) < 0.12 and abs(pri / prit - 1) < 0.01, 'pulsed pw %.2f us pri %.1f us' % (pw * 1e6, pri * 1e6))
        print('consistency      pulsed @%2.0fM  pw %.2f us (truth %.2f), PRI %.2f us (truth %.2f)' % (rate / 1e6, pw * 1e6, pwt * 1e6, pri * 1e6, prit * 1e6))
    # envelope statistics a characteriser would use (informational): kurtosis of the channel-filtered envelope power
    print('envelope stats   kurtosis of the amplitude |y| (channel filtered, 20 dB in-band SNR, burst samples only); noise only = 3.25 (Rayleigh)')
    row = []
    for kind, kw, rate in (('cw', {}, 2e6), ('nfm', {}, 2e6), ('fsk2', {}, 2e6), ('bpsk', {}, 2e6), ('qpsk', {}, 2e6), ('am', {}, 2e6),
                           ('ook', {}, 2e6), ('ofdm', {}, 2e6), ('lora', {}, 2e6), ('gfsk', {}, 10e6)):
        r = gen.make(kind, rate, 0.5, 20.0, seed=14, **kw)
        t = r['truth']
        y = _chan(r['iq'], rate, t['center_hz'], max(0.55 * t['occupied_bw_hz'], 2e3))
        p = np.abs(y)
        if t['on_intervals'] and t['on_fraction'] < 0.98:
            m = np.zeros(len(p), bool)
            for a, b in t['on_intervals']:
                d = 0.15 * (b - a)
                m[int((a + d) * rate):int((b - d) * rate)] = True
            p = p[m]
        k = float(np.mean((p - p.mean()) ** 4) / np.var(p) ** 2)
        row.append('%s %.2f' % (kind, k))
    n = gen.make('noise', 2e6, 0.5, 0.0, seed=1)
    y = _chan(n['iq'], 2e6, 0.0, 100e3)
    p = np.abs(y)
    row.append('noise(100k) %.2f' % float(np.mean((p - p.mean()) ** 4) / np.var(p) ** 2))
    print('                 ' + ', '.join(row))


def run_timing(rep, long):
    print('timing           0.5 s @ 10 Msps (generation incl. int8 quantisation):')
    worst = 0.0
    parts = []
    for name, kind, kw, rates, snrs, csnr in MATRIX:
        if kind == 'multi' or name.endswith('narrow') or name in ('nfm-tone', 'am-voice', 'ook-manchester', 'lora-sf9', 'pulsed-lfm', 'pulsed-fast'):
            continue
        kwr = dict(kw)
        if kind == 'pulsed':
            kwr['pw_s'] = 2e-6
        t0 = time.time()
        gen.make(kind, 10e6, 0.5, 10.0, seed=1, **kwr)
        dt = time.time() - t0
        worst = max(worst, dt)
        parts.append('%s %.2fs' % (kind, dt))
    print('   ' + ', '.join(parts))
    rep.check(worst < 4.0, 'slow generation: %.2f s' % worst)
    rep.note('time_0.5s_10M_max_s', worst)
    if long:
        import resource
        for kind in ('cw', 'ofdm', 'gfsk'):
            t0 = time.time()
            r = gen.make(kind, 10e6, 5.0, 10.0, seed=1)
            print('   5 s @ 10 Msps %-5s %.1f s, ci8 %d MB, peak RSS %.0f MB, clip %.5f'
                  % (kind, time.time() - t0, len(r['ci8']) // 10 ** 6, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3,
                     r['truth']['clip_fraction']))
            del r


def main(argv):
    quick = '--quick' in argv
    long = '--long' in argv
    rep = Report()
    t0 = time.time()
    results = run_matrix(rep, quick)
    run_noise(rep)
    run_special(rep)
    run_consistency(rep)
    run_timing(rep, long)
    print()
    print('worst-case numbers:')
    for k, v in sorted(rep.worst.items()):
        print('   %-26s %.4g' % (k, v))
    print('checks failed: %d   (%.0f s)' % (len(rep.fail), time.time() - t0))
    for f in rep.fail:
        print('  -', f)
    if '--json' in argv:
        path = argv[argv.index('--json') + 1]
        with open(path, 'w') as fh:
            json.dump(dict(worst=rep.worst, failures=rep.fail, rows=results), fh, indent=1, default=float)
    print('RESULT:', 'PASS' if not rep.fail else 'FAIL')
    return 0 if not rep.fail else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

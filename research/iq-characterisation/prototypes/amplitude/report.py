"""Text tables from a bench jsonl.  python3 report.py runs/eval.jsonl [section ...]
Sections: presence duty shape silent auc labels chip otsu raw"""
import sys
import math
import numpy as np
from tab import load, col, sel, auc, med, q
import decide

SNRS = [0, 5, 10, 15, 20, 30]
CE_SC = ['cw', 'nfm_voice', 'nfm_tone', 'wfm', 'fsk2_rect', 'fsk2_gauss']
BURSTY_CE = ['ook_pwm', 'ook_man', 'gfsk', 'lora7', 'lora9', 'hopper']
ALL_SIG = ['cw', 'am_tone', 'am_voice', 'nfm_voice', 'nfm_tone', 'wfm', 'ook_pwm', 'ook_man', 'fsk2_rect', 'fsk2_gauss', 'gfsk', 'bpsk',
           'qpsk', 'lora7', 'lora9', 'ofdm_wide', 'ofdm_narrow', 'hopper', 'pulsed', 'pulsed_fast', 'pulsed_lfm', 'multi_cw_nfm', 'multi_ook_nfm']


def present(r):
    in_win = abs(r.get('line_f', 1e12) - r.get('prior_fc', 0.0)) <= r.get('prior_bw', 1e12)
    p_psd = r.get('ch_src') in ('psd', 'line', 'psd+prior')
    p_line = r.get('line_over_thr_db', -9) > 0 and in_win
    p_time = bool(r.get('det_any', 0.0))
    p_car = r.get('car_carrier_over_thr_db', -9) > 0 and r.get('ch_src') is not None
    return p_psd, p_line or p_car, p_time


def sec_presence(rows):
    print('\n=== PRESENCE (any of: psd cluster in prior window / spectral line / burst energy), % of captures flagged')
    print('%-14s %-7s ' % ('scenario', 'duty') + ' '.join('%5d' % s for s in SNRS))
    for sc in ['noise2', 'noise10'] + ALL_SIG:
        for om in ('native', 'gated'):
            rr = sel(rows, scen=sc, onmode=om)
            if not rr: continue
            if sc.startswith('noise') and om == 'gated': continue
            line = []
            for s in SNRS:
                r2 = sel(rr, snr=s) if not sc.startswith('noise') else rr
                if not r2: line.append('  n/a'); continue
                pr = [any(present(r)) for r in r2]
                line.append('%5.0f' % (100 * np.mean(pr)))
            print('%-14s %-7s ' % (sc, om) + ' '.join(line))


def sec_duty(rows):
    print('\n=== SEGMENTATION vs truth on-intervals (blind): median |duty err| (absolute, in duty units) / event recall / precision / n_det:n_truth')
    for om in ('native', 'gated'):
        print('-- %s' % om)
        print('%-14s ' % 'scenario' + ' | '.join('snr %-2d  derr  rec  prec nrat' % s for s in SNRS))
        for sc in ALL_SIG:
            if sc.startswith('multi'): continue
            cells = []
            for s in SNRS:
                rr = [r for r in sel(rows, scen=sc, onmode=om, snr=s) if r.get('has_channel')]
                if not rr: cells.append('     n/a                  '); continue
                de = np.abs(col(rr, 'duty') - col(rr, 't_duty'))
                rec = col(rr, 'ev_recall'); pre = col(rr, 'ev_precision')
                nr = col(rr, 'ev_n_det') / np.maximum(col(rr, 'ev_n_truth'), 1)
                cells.append('     %5.3f %4.2f %4.2f %4.2f' % (med(de), med(rec), med(pre), med(nr)))
            print('%-14s ' % sc + ' | '.join(cells))


def sec_shape(rows):
    print('\n=== ON-STATE ENVELOPE SHAPE: median noise-compensated kurtosis m4_sig (CE=1, QPSK-RRC~1.2, Gaussian=2), blind mask / oracle mask+band / channel SNR dB')
    for om in ('native', 'gated'):
        print('-- %s' % om)
        print('%-14s ' % 'scenario' + ' | '.join('snr %-2d  blind orcl snrch' % s for s in SNRS))
        for sc in ALL_SIG:
            cells = []
            for s in SNRS:
                rr = sel(rows, scen=sc, onmode=om, snr=s)
                if not rr: cells.append('        n/a            '); continue
                cells.append('      %5.2f %5.2f %5.1f' % (med(col(rr, 'on_m4_sig')), med(col(rr, 'o_on_m4_sig')), med(col(rr, 'on_snr_ch_db'))))
            print('%-14s ' % sc + ' | '.join(cells))


def theory_whole(d, snr_lin, m4s=1.0):
    return (2 + d * (4 * snr_lin + m4s * snr_lin ** 2)) / (1 + d * snr_lin) ** 2


def sec_silent(rows):
    print('\n=== SILENT PORTIONS: whole-capture m4 vs active-only m4 (blind mask) vs oracle active-only, gated captures, SNR 15/30 dB')
    print('%-12s %-4s | %-8s %-8s %-8s %-8s | %-8s %-10s' % ('scenario', 'snr', 'duty_t', 'm4_whole', 'm4_act', 'm4_orcl', 'rfs_raw', 'theory_whl'))
    for sc in ['cw', 'nfm_voice', 'fsk2_gauss', 'ook_pwm', 'gfsk', 'lora7', 'qpsk', 'ofdm_narrow', 'ofdm_wide', 'hopper', 'am_voice', 'pulsed']:
        for s in (15, 30):
            rr = sel(rows, scen=sc, onmode='gated', snr=s)
            if not rr: continue
            snr_lin = 10 ** (med(col(rr, 'o_on_snr_ch_db')) / 10.0)
            d = med(col(rr, 't_duty'))
            print('%-12s %-4d | %-8.3f %-8.3f %-8.3f %-8.3f | %-8.2f %-10.3f' % (sc, s, d, med(col(rr, 'all_m4')), med(col(rr, 'on_m4')), med(col(rr, 'o_on_m4')),
                                                                                 med(col(rr, 'raw_rfs_kurt')), theory_whole(d, snr_lin)))


def best_acc(pos, neg):
    """best single-threshold accuracy (balanced) between two samples, either orientation"""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 3 or len(neg) < 3: return float('nan')
    a = auc(pos, neg)
    return max(a, 1 - a)


def sec_auc(rows):
    feats = [('on_m4_sig', 'm4_sig(blind on)'), ('o_on_m4_sig', 'm4_sig(oracle on)'), ('on_m4', 'm4 raw on-state'), ('on_amp_kurt', 'amp kurt(on)'),
             ('on_cv', 'CV(on)'), ('on_papr_db', 'PAPR(on)'), ('on_iq_kurt', 'IQ kurt(on)'), ('all_m4', 'm4 whole capture'), ('raw_rfs_kurt', 'RFS raw |iq| kurt')]
    for name, A_, B_ in [('CE vs GAUSS (ofdm_narrow+ofdm_wide)', ['nfm_voice', 'nfm_tone', 'fsk2_rect', 'fsk2_gauss', 'wfm', 'cw'], ['ofdm_narrow', 'ofdm_wide']),
                         ('CE vs LIN (qpsk+bpsk)', ['nfm_voice', 'nfm_tone', 'fsk2_rect', 'fsk2_gauss', 'wfm', 'cw'], ['qpsk', 'bpsk']),
                         ('LIN(qpsk,bpsk) vs GAUSS', ['qpsk', 'bpsk'], ['ofdm_narrow', 'ofdm_wide']),
                         ('AM(tone+voice) vs GAUSS', ['am_tone', 'am_voice'], ['ofdm_narrow', 'ofdm_wide']),
                         ('NOISE vs CE continuous (whole-capture m4 in prior channel)', ['noise2'], ['nfm_voice', 'nfm_tone', 'fsk2_rect', 'fsk2_gauss', 'wfm'])]:
        print('\n=== AUC %s (native, per SNR; 0.5 = useless, 1.0 = perfect; orientation-free)' % name)
        print('%-20s ' % 'feature' + ' '.join('%6d' % s for s in SNRS))
        for k, lab in feats:
            cells = []
            for s in SNRS:
                if A_ == ['noise2']:
                    pa = col([r for r in rows if r['scen'] == 'noise2'], k)
                else:
                    pa = np.concatenate([col(sel(rows, scen=a, onmode='native', snr=s), k) for a in A_])
                pb = np.concatenate([col(sel(rows, scen=b, onmode='native', snr=s), k) for b in B_])
                cells.append('%6.2f' % best_acc(pa, pb))
            print('%-20s ' % lab + ' '.join(cells))


def sec_labels(rows):
    for use_o in (False, True):
        print('\n=== DECISION LAYER (%s mask/band) : per scenario at each SNR the most frequent label, and accuracy among answered, coverage' % ('ORACLE' if use_o else 'BLIND'))
        for om in ('native', 'gated'):
            print('-- %s' % om)
            for sc in ['noise2', 'noise10'] + [x for x in ALL_SIG if not x.startswith('multi')]:
                if sc.startswith('noise') and om == 'gated': continue
                cells = []
                for s in SNRS:
                    rr = sel(rows, scen=sc, onmode=om, snr=s) if not sc.startswith('noise') else sel(rows, scen=sc)
                    if not rr: cells.append('n/a'); continue
                    res = [decide.decide(r, use_oracle=use_o) for r in rr]
                    tl = [decide.true_label(r) for r in rr]
                    ok = sum(1 for a, t in zip(res, tl) if a['label'] == t)
                    unk = sum(1 for a in res if a['label'] == 'UNKNOWN')
                    ans = len(rr) - unk
                    from collections import Counter
                    top = Counter(a['label'] for a in res).most_common(1)[0][0]
                    cells.append('%s %d%%/%d%%' % (top[:10], 100 * ok // len(rr), 100 * ans // len(rr)))
                print('%-14s ' % sc + ' | '.join('%-18s' % c for c in cells))


def sec_chip(rows):
    print('\n=== SYMBOL / CHIP RATE HINT from on/off run lengths (blind), OOK only: median rel. error, % within 10 %, median quality; and quality of non-OOK classes')
    for sc in ('ook_pwm', 'ook_man'):
        for om in ('native', 'gated'):
            cells = []
            for s in SNRS:
                rr = [r for r in sel(rows, scen=sc, onmode=om, snr=s) if r.get('has_channel')]
                if not rr: cells.append('   n/a'); continue
                e = np.abs(col(rr, 'chip_hz') / col(rr, 't_symrate') - 1)
                cells.append('%4.0f%%/%2.0f%%/q%.2f' % (100 * med(e), 100 * np.nanmean(e < 0.1), med(col(rr, 'chip_q'))))
            print('%-9s %-7s ' % (sc, om) + ' | '.join(cells))
    print('chip quality (median) for other classes at 20 dB (native / gated): high quality in a non-OOK class = false OOK hint')
    for sc in ['cw', 'nfm_voice', 'am_tone', 'am_voice', 'gfsk', 'lora7', 'hopper', 'pulsed', 'ofdm_wide']:
        a = med(col(sel(rows, scen=sc, onmode='native', snr=20), 'chip_q')); b = med(col(sel(rows, scen=sc, onmode='gated', snr=20), 'chip_q'))
        print('  %-12s native q %.2f   gated q %.2f' % (sc, a, b))


def sec_otsu(rows):
    print('\n=== ON-FRACTION: HMM (noise-referenced) vs Otsu split of log smoothed power, median |error| vs truth duty (gated, 10-30 %)')
    print('%-12s ' % 'scenario' + ' | '.join('snr %-2d hmm  otsu' % s for s in SNRS))
    for sc in ['cw', 'nfm_voice', 'ook_pwm', 'gfsk', 'lora7', 'ofdm_wide', 'hopper', 'pulsed', 'qpsk']:
        cells = []
        for s in SNRS:
            rr = [r for r in sel(rows, scen=sc, onmode='gated', snr=s) if r.get('has_channel')]
            if not rr: cells.append('        n/a      '); continue
            e1 = np.abs(col(rr, 'duty') - col(rr, 't_duty')); e2 = np.abs(col(rr, 'otsu_duty') - col(rr, 't_duty'))
            cells.append('     %5.3f %5.3f' % (med(e1), med(e2)))
        print('%-12s ' % sc + ' | '.join(cells))
    # on noise
    rr = sel(rows, scen='noise2')
    print('Otsu on noise-only: it ALWAYS splits: median otsu duty %.2f, eta %.2f (HMM duty %.3f)' % (med(col(rr, 'otsu_duty')), med(col(rr, 'otsu_eta')), med(col(rr, 'duty'))))


def sec_raw(rows):
    print('\n=== RAW WIDEBAND "RF-Sentinel" kurtosis of |iq| (alert > 8) : median / fraction > 8, per scenario & SNR (gated=on_fraction 10-30 %)')
    print('%-14s %-7s ' % ('scenario', 'duty') + ' '.join('%11d' % s for s in SNRS))
    for sc in ['noise2', 'noise10'] + ALL_SIG:
        for om in ('native', 'gated'):
            if sc.startswith('noise') and om == 'gated': continue
            cells = []
            for s in SNRS:
                rr = sel(rows, scen=sc, onmode=om, snr=s) if not sc.startswith('noise') else sel(rows, scen=sc)
                if not rr: cells.append('        n/a'); continue
                v = col(rr, 'raw_rfs_kurt')
                cells.append('%5.2f/%3.0f%%' % (med(v), 100 * np.mean(v > 8)))
            print('%-14s %-7s ' % (sc, om) + ' '.join(cells))


SECS = dict(presence=sec_presence, duty=sec_duty, shape=sec_shape, silent=sec_silent, auc=sec_auc, labels=sec_labels, chip=sec_chip, otsu=sec_otsu, raw=sec_raw)

def _bucket(snr):
    return '0-5' if snr <= 5 else ('10-15' if snr <= 15 else '20-30')


def sec_confusion(rows):
    from collections import Counter, defaultdict
    labs = ['NOISE', 'CARRIER', 'CE_CONT', 'CE_BURSTY', 'LIN_CONT', 'LIN_BURSTY', 'GAUSS_CONT', 'GAUSS_BURSTY', 'PULSED', 'UNKNOWN']
    for use_o in (False, True):
        for bk in ('0-5', '10-15', '20-30'):
            conf = defaultdict(Counter)
            for r in rows:
                t = decide.true_label(r)
                if t is None: continue
                if not r['scen'].startswith('noise') and _bucket(r['snr']) != bk: continue
                if r['scen'].startswith('noise') and bk != '0-5': continue
                d = decide.decide(r, use_oracle=use_o)
                conf[t][d['label']] += 1
            print('\n=== CONFUSION (%s) in-band SNR %s dB: rows = truth, %% of captures' % ('ORACLE mask+band' if use_o else 'BLIND', bk))
            print('%-13s %5s ' % ('truth', 'n') + ' '.join('%7s' % l[:7] for l in labs))
            tot_ok = tot_n = tot_unk = 0
            for t in labs[:-1]:
                if t not in conf: continue
                n = sum(conf[t].values())
                print('%-13s %5d ' % (t, n) + ' '.join('%6.0f%%' % (100 * conf[t][l] / n) for l in labs))
                tot_n += n; tot_ok += conf[t][t]; tot_unk += conf[t]['UNKNOWN']
            ans = tot_n - tot_unk
            print('overall: correct %.1f%% | unknown %.1f%% | wrong %.1f%% | accuracy among answered %.1f%%' % (
                100 * tot_ok / tot_n, 100 * tot_unk / tot_n, 100 * (tot_n - tot_ok - tot_unk) / tot_n, 100 * tot_ok / max(ans, 1)))


def sec_calib(rows):
    print('\n=== CONFIDENCE CALIBRATION (blind, all SNR, answered only): accuracy per reported-confidence bin')
    res = []
    for r in rows:
        t = decide.true_label(r)
        if t is None: continue
        d = decide.decide(r)
        if d['label'] == 'UNKNOWN': continue
        res.append((d['conf'], d['label'] == t, r['snr']))
    c = np.array([x[0] for x in res]); ok = np.array([x[1] for x in res], float)
    for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.999), (0.999, 1.01)]:
        m = (c >= lo) & (c < hi)
        if m.sum(): print('  conf in [%.3f,%.3f): n=%5d  accuracy %.1f%%' % (lo, hi, m.sum(), 100 * ok[m].mean()))


def first_snr(vals_by_snr, thr):
    """lowest SNR from which the criterion holds at that SNR and all higher ones"""
    good = None
    for s in sorted(vals_by_snr, reverse=True):
        v = vals_by_snr[s]
        if v == v and v >= thr: good = s
        else: break
    return good


def sec_minsnr(rows):
    print('\n=== LOWEST IN-BAND SNR (dB) AT WHICH EACH FEATURE WORKS (criterion holds there and at every higher grid SNR)')
    print('criteria: presence >=90 % of captures flagged | duty |err|<=0.05 abs for >=90 % | shape m4_sig within +-0.15 of the 30 dB class value for >=90 %')
    print('%-14s %-7s | %-9s %-9s %-9s %-9s' % ('scenario', 'duty', 'presence', 'duty(bl)', 'shape(bl)', 'shape(orc)'))
    for sc in ALL_SIG:
        for om in ('native', 'gated'):
            pres, dut, shp, shpo = {}, {}, {}, {}
            ref = med(col(sel(rows, scen=sc, onmode=om, snr=30), 'o_on_m4_sig'))
            for s in SNRS:
                rr = sel(rows, scen=sc, onmode=om, snr=s)
                if not rr: continue
                pres[s] = np.mean([any(present(r)) for r in rr])
                rc = [r for r in rr if r.get('has_channel')]
                dut[s] = np.mean(np.abs(col(rc, 'duty') - col(rc, 't_duty')) <= 0.05) if rc else float('nan')
                shp[s] = np.mean(np.abs(col(rr, 'on_m4_sig') - ref) <= 0.15) if ref == ref else float('nan')
                shpo[s] = np.mean(np.abs(col(rr, 'o_on_m4_sig') - ref) <= 0.15) if ref == ref else float('nan')
            f = lambda d: ('%d' % first_snr(d, 0.9) if first_snr(d, 0.9) is not None else 'never')
            print('%-14s %-7s | %-9s %-9s %-9s %-9s' % (sc, om, f(pres), f(dut), f(shp), f(shpo)))


def sec_signatures(rows):
    print('\n=== CLASS SIGNATURES at 30 dB with oracle mask/band (on-state, noise-compensated): m4_sig | amp kurtosis (central) | CV | PAPR dB | I/Q kurtosis | non-clipped captures only')
    for sc in ALL_SIG:
        rr = [r for r in sel(rows, scen=sc, snr=30) if r.get('raw_clip_frac', 0) <= 1e-3]
        if not rr: rr = sel(rows, scen=sc, snr=20)
        print('%-14s n=%-3d  %5.2f | %5.2f | %5.2f | %5.1f | %5.2f' % (sc, len(rr), med(col(rr, 'o_on_m4_sig')), med(col(rr, 'o_on_amp_kurt')), med(col(rr, 'o_on_cv')), med(col(rr, 'o_on_papr_db')), med(col(rr, 'o_on_iq_kurt'))))
    rr = sel(rows, scen='noise2')
    print('%-14s n=%-3d  (m4 %5.2f) | %5.2f | %5.2f | %5.1f | %5.2f' % ('noise', len(rr), med(col(rr, 'all_m4')), med(col(rr, 'all_amp_kurt')), med(col(rr, 'all_cv')), med(col(rr, 'all_papr_db')), med(col(rr, 'all_iq_kurt'))))


SECS.update(confusion=sec_confusion, calib=sec_calib, minsnr=sec_minsnr, sig=sec_signatures)


if __name__ == '__main__':
    rows = [r for r in load(sys.argv[1]) if 'error' not in r]
    which = sys.argv[2:] or list(SECS)
    for w in which:
        SECS[w](rows)

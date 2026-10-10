p = open('pulse.py').read()
p = p.replace("SCALES = (4, 16, 64)               # box-car lengths (samples), factor 4 apart: <= ~2 dB mismatch loss",
              "SCALES = (4, 16, 64, 256, 1024)   # box-car lengths (samples), factor 4 apart: <= ~2 dB mismatch loss")
p = p.replace('''        flag = S > thr
        if not flag.any():
            continue
        st, en = _runs(flag)''', '''        flag = S > thr
        if not flag.any():
            continue
        if len(flag) > 3:                       # close short holes so a long burst is one event
            flag = np.convolve(flag.astype(np.int8), np.ones(5, np.int8), mode='same') > 0
        st, en = _runs(flag)''')
# decision: new signature using stack width
a = p.index('def decide_pulsed(')
b = p.index('# ----------------------------------------------------------------------------------------------\n# intra-pulse chirp')
new = '''def decide_pulsed(events, rate, pri, width_s, cfg):
    """Return (flag, info).  `width_s`: -3 dB width of the stacked pulse profile (or None).  The per-event width
    spread is only judged on the strong events (peak > 8 dB over the noise), where it is measurable."""
    n = len(events)
    info = dict(n_events=n)
    if n < cfg['n_min'] or not pri.get('pri'):
        info['why'] = 'too few events or no PRI'
        return False, info
    w = np.array([(e['t1'] - e['t0']) / rate for e in events])
    strong = np.array([e['snr'] >= 6.3 for e in events])
    wm = float(width_s) if width_s else float(np.median(w))
    duty = wm / pri['pri']
    info.update(width=wm, pri=pri['pri'], jitter=pri['jitter'], frac_multiple=pri['frac_multiple'], duty=duty,
                n_cluster=pri.get('n_cluster', 0), n_strong=int(strong.sum()))
    cv = None
    if strong.sum() >= 20:
        ws = w[strong]
        med = float(np.median(ws))
        cv = float(np.median(np.abs(ws - med)) / max(med, 1e-12) * 1.4826)
        info['width_cv'] = cv
    reasons = []
    if pri.get('n_cluster', 0) < cfg['n_cluster_min']:
        reasons.append('few in-train spacings')
    if pri['frac_multiple'] < cfg['multiple_min']:
        reasons.append('PRI not regular')
    if (pri['jitter'] or 0) > cfg['jitter_max']:
        reasons.append('PRI jitter')
    if duty > cfg['duty_max']:
        reasons.append('duty too high')
    if cv is not None and cv > cfg['cv_max']:
        reasons.append('width spread')
    info['why'] = ', '.join(reasons)
    return (len(reasons) == 0), info


'''
p = p[:a] + new + p[b:]
open('pulse.py', 'w').write(p)

a = open('analyze.py').read()
i = a.index('def pulse_stage(')
j = a.index('def label_of(')
new = '''def pulse_stage(events, rate, cfg, src=None, mean=0j, noise=None):
    out = dict(n_events=len(events))
    if len(events) < 3:
        out['flag'] = False
        out['why'] = 'fewer than 3 events'
        return out
    t0s = np.array([e['t0'] for e in events]) / rate
    pri = pl.estimate_pri(t0s, rate)
    out['pri'] = pri
    width_s = None
    regular = pri.get('pri') and pri['frac_multiple'] >= 0.5 and pri.get('n_cluster', 0) >= 5
    if regular and src is not None:
        st = pl.stack_profile(src, mean, events, noise)
        if st is not None:
            width_s = st['width_samples'] / rate
            out['width_s'] = width_s
            out['stack_snr_db'] = 10 * math.log10(max(st['snr'], 1e-9))
            out['stack_n'] = st['n']
    flag, info = pl.decide_pulsed(events, rate, pri, width_s, cfg['pulse'])
    out.update(flag=bool(flag), info=info)
    if flag and src is not None:
        out['chirp_pulse'] = pl.intrapulse_slope(src, mean, events, rate)
    return out


'''
a = a[:i] + new + a[j:]
open('analyze.py', 'w').write(a)

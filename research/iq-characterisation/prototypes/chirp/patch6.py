p = open('pulse.py').read()
# stack iters=2
p = p.replace("def stack_profile(src, mean, events, noise, max_events=600, iters=3):", "def stack_profile(src, mean, events, noise, max_events=600, iters=2):")
# strong events: 15 dB
p = p.replace("strong = np.array([e['snr'] >= 6.3 for e in events])", "strong = np.array([e['snr'] >= 31.6 for e in events])")
p = p.replace("# PRI\n", "# PRI\n", 1)
add = '''def completeness(t, pri, split=4.5):
    """Share of the pulses expected on the PRI grid inside each train (a train ends at a gap > split*PRI) that
    were actually detected.  Low values mean only the noise-boosted pulses were found (selection bias: the
    stacked width and the PRI are then optimistic)."""
    t = np.sort(np.asarray(t, np.float64))
    if len(t) < 2 or not pri:
        return 0.0
    gaps = np.diff(t)
    brk = np.flatnonzero(gaps > split * pri)
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk + 1, [len(t)]])
    exp = 0.0
    det = 0.0
    for a, b in zip(starts, ends):
        n = b - a
        if n < 2:
            continue
        exp += (t[b - 1] - t[a]) / pri + 1.0
        det += n
    return float(min(1.0, det / exp)) if exp > 0 else 0.0


'''
p = p.replace("def decide_pulsed(", add + "def decide_pulsed(", 1)
open('pulse.py', 'w').write(p)
a = open('analyze.py').read()
a = a.replace("    out['pri'] = pri\n    width_s = None", "    out['pri'] = pri\n    if pri.get('pri'):\n        out['completeness'] = pl.completeness(t0s, pri['pri'])\n    width_s = None")
open('analyze.py', 'w').write(a)

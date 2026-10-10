g = open('gated.py').read()
i = g.index('def occupied(')
g = g[:i] + '''def occupied(freqs, psd, nseg, floor, fc, win_hz=370e3, smooth_hz=6e3):
    """Occupied band of the strongest spectral component within +-win_hz of fc, from the excess PSD (psd - floor).
    The component is the connected set of smoothed bins above max(4 sigma, 25 % of the median excess of the bins
    above 4 sigma) that contains the maximum (gaps of <= 3 bins are bridged, so spectral horns / ripple do not split a
    flat-topped signal).  Returns dict(measurable, bw, centre, lo, hi, plateau_db, n_other) - `n_other` counts the
    other separate components in the window."""
    df = freqs[1] - freqs[0]
    sel = np.abs(freqs - fc) <= win_hz
    f, ex = freqs[sel], (psd - floor)[sel]
    s = max(1, int(round(smooth_hz / df)))
    exs = np.convolve(ex, np.ones(s) / s, mode='same')
    sd = floor / np.sqrt(max(nseg / 1.8, 1.0) * s)
    big = exs > 4.0 * sd
    out = dict(measurable=False)
    if big.sum() < 3:
        return out
    med = float(np.median(exs[big]))
    thr = max(4.0 * sd, 0.25 * med)
    mask = exs > thr
    bridged = np.convolve(mask.astype(np.int8), np.ones(7, np.int8), mode='same') >= 1
    bridged = np.convolve(bridged.astype(np.int8), np.ones(7, np.int8), mode='same') >= 7        # close, not dilate
    mask = mask | bridged & (np.convolve(mask.astype(np.int8), np.ones(7, np.int8), mode='same') >= 2)
    wide = np.convolve(exs, np.ones(9) / 9.0, mode='same')
    k = int(np.argmax(np.where(mask, wide, -1e30)))
    if not mask[k]:
        return out
    lo = k
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = k
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1
    st = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])) == 1)
    out.update(measurable=True, bw=float((hi - lo + 1) * df), centre=float(0.5 * (f[lo] + f[hi])), lo=float(f[lo]),
               hi=float(f[hi]), plateau_db=float(10 * np.log10(1 + med / floor)), n_other=int(len(st) - 1))
    return out
'''
open('gated.py', 'w').write(g)
a = open('analyze.py').read()
a = a.replace("band = gated.occupied(g[0], g[1], g[2], floor, cand['fc'], cand['bw'])", "band = gated.occupied(g[0], g[1], g[2], floor, cand['fc'])")
a = a.replace("            r = band['bw3'] / cand['bw']\n            bw_final = _nearest_bw(band['bw3'])\n            if bw_final is None or r < cfg['bw_ratio_min'] * 0.5 and False:\n                pass\n", "            bw_final = _nearest_bw(band['bw'])\n")
a = a.replace("'occupied bandwidth %.0f kHz is not a LoRa bandwidth' % (band['bw3'] / 1e3)", "'occupied bandwidth %.0f kHz is not a LoRa bandwidth' % (band['bw'] / 1e3)")
open('analyze.py', 'w').write(a)

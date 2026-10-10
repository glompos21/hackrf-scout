s = open('css.py').read()
a = s.index('TRIM = 7')
b = s.index('def lag_blocks(')
new = '''def bg_bins(L, q_div=8, near=3, far_hi=10, far_lo=7):
    """Background bins around a line at k0 = L/q_div: k0+near .. k0+far_hi and k0-far_lo .. k0-near (never bin 0
    or the DC leakage k < 2).  A LOCAL background: data-modulated signals (slow FSK/OOK/AM/NFM) have a coloured
    z spectrum that falls off with frequency, and a far-away background then makes their ordinary spectrum
    look like a line."""
    k0 = L // q_div
    hi = list(range(k0 + near, min(L // 2, k0 + far_hi) + 1))
    lo = [k for k in range(k0 - far_lo, k0 - near + 1) if k >= 2]
    return np.array(lo + hi)


def block_stats(seg, L=64, q_div=8, trim=1, near=3, far_hi=10, far_lo=7):
    """seg: (nb, L) complex decimated-z blocks.  Returns (R_up, R_dn): power in the 3 bins around the +-line
    (bin L/q_div) over 3 x trimmed-mean power of the LOCAL background bins (largest `trim` dropped)."""
    k0 = L // q_div
    w = np.hanning(L).astype(np.float32)
    seg = seg - seg.mean(1, keepdims=True)
    F = np.fft.fft(seg * w[None, :], axis=1)
    P = F.real * F.real + F.imag * F.imag
    idx_u = bg_bins(L, q_div, near, far_hi, far_lo)
    idx_d = (L - idx_u) % L
    res = []
    for k_line, idx in ((k0, idx_u), (L - k0, idx_d)):
        bgv = P[:, idx]
        nbg = bgv.shape[1]
        if trim > 0:
            bg = np.partition(bgv, nbg - trim - 1, axis=1)[:, :nbg - trim].mean(1)
        else:
            bg = bgv.mean(1)
        res.append((P[:, k_line - 1:k_line + 2].sum(1) / (3.0 * bg + 1e-30)).astype(np.float32))
    return res[0], res[1]


'''
s = s[:a] + new + s[b:]
open('css.py', 'w').write(s)

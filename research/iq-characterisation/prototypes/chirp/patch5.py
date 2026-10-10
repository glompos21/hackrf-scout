s = open('css.py').read()
a = s.index('def bg_bins(')
b = s.index('def block_stats(')
new = '''def bg_bins(L, q_div=8, near=3, hi_mult=2.0):
    """Background bins around a line at k0 = L/q_div: upper side k0+near .. hi_mult*k0 + near-ish, lower side
    2 .. k0-near (never bin 0 or the DC leakage bin 1).  A LOCAL background: data-modulated signals (slow FSK/OOK/AM/
    NFM) have a coloured z spectrum that falls off with frequency, and a far-away background then makes their
    ordinary spectrum look like a line."""
    k0 = L // q_div
    hi = list(range(k0 + near, min(L // 2, int(round(hi_mult * k0)) + near * 3) + 1))
    lo = [k for k in range(2, k0 - near + 1)]
    return np.array(lo + hi)


'''
s = s[:a] + new + s[b:]
s = s.replace("def block_stats(seg, L=64, q_div=8, trim=1, near=3, far_hi=10, far_lo=7):", "def block_stats(seg, L=64, q_div=8, trim=1, near=3, hi_mult=2.0):")
s = s.replace("idx_u = bg_bins(L, q_div, near, far_hi, far_lo)", "idx_u = bg_bins(L, q_div, near, hi_mult)")
open('css.py', 'w').write(s)
n = open('nulls.py').read()
n = n.replace("for L in (64,):", "for L in (64, 128):")
n = n.replace("r = mc(L, int(nm * 1e6), seed=L)", "r = mc(L, int(nm * 1e6 * (1.0 if L == 64 else 0.5)), seed=L)")
open('nulls.py', 'w').write(n)

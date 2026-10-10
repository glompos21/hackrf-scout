a = open('analyze.py').read()
# null threshold function with k
i = a.index('_NULL = None')
j = a.index('# ----------------------------------------------------------------------------------------------\n# the analysis')
new = '''_NULL = {}


def lag_null_threshold(L, p, k=1):
    """Threshold on min(R) over k consecutive blocks (same direction) with null exceedance probability p, from the
    Monte-Carlo table results/null_lagk_<L>.json by log-log interpolation / extrapolation (power-law tail)."""
    if L not in _NULL:
        _NULL[L] = json.load(open(os.path.join(HERE, 'results', 'null_lagk_%d.json' % L)))
    tab = _NULL[L][str(k)]
    ps = np.array(sorted([float(q) for q in tab if q != 'n'], reverse=True))
    ts = np.array([tab['%g' % q] for q in ps])
    keep = ps * tab['n'] >= 30                      # levels with enough Monte-Carlo exceedances
    ps, ts = ps[keep], ts[keep]
    lp, lt = np.log(ps), np.log(ts)
    if p >= ps[0]:
        return float(ts[0])
    if p <= ps[-1]:
        m = min(3, len(ps) - 1)
        slope = (lt[-1] - lt[-1 - m]) / (lp[-1] - lp[-1 - m])
        return float(math.exp(lt[-1] + slope * (math.log(p) - lp[-1])))
    return float(math.exp(np.interp(math.log(p), lp[::-1], lt[::-1])))


def kmin(x, k):
    """Running minimum over k consecutive elements (length n-k+1)."""
    if k <= 1:
        return x
    out = x[:len(x) - k + 1].copy()
    for j in range(1, k):
        out = np.minimum(out, x[j:len(x) - k + 1 + j])
    return out


'''
a = a[:i] + new + a[j:]
# in chirp_stage: use k-adjacent statistic
a = a.replace('''    p_block = cfg['p_capture'] / max(n_trials, 1)
    T = lag_null_threshold(L, p_block)
    passing = []
    mx = 0.0
    for ci, bw, fc, sf, lb in lag:
        up, dn = lb['up'], lb['dn']
        mx = max(mx, float(up.max()), float(dn.max()))
        for name, main, opp in (('up', up, dn), ('down', dn, up)):
            ok = (main > T) & (main > cfg['asym'] * opp)
            if ok.any():
                k = int(np.argmax(np.where(ok, main, 0)))
                hit = np.flatnonzero(ok)
                passing.append(dict(ci=ci, bw=bw, fc=fc, sf=sf, dir=name, R=float(main[k]), score=float(main[k] / T),
                                    n_hits=int(len(hit)), n_blocks=int(len(main)), block=k, start=int(lb['start'][k]),
                                    span=int(lb['span']), hit_idx=hit, main=main, lb=lb,
                                    adjacent=bool(len(hit) >= 2 and np.any(np.diff(hit) <= 2))))''', '''    kadj = cfg.get('k_adj', 2)
    p_block = cfg['p_capture'] / max(n_trials, 1)
    T = lag_null_threshold(L, p_block, kadj)
    passing = []
    mx = 0.0
    for ci, bw, fc, sf, lb in lag:
        up, dn = lb['up'], lb['dn']
        for name, main, opp in (('up', up, dn), ('down', dn, up)):
            mm = np.where(main > cfg['asym'] * opp, main, 0.0)          # mirror-line (asymmetry) test per block
            mk = kmin(mm, kadj)
            if len(mk) == 0:
                continue
            mx = max(mx, float(mk.max()))
            ok = mk > T
            if ok.any():
                k = int(np.argmax(mk))
                hit = np.flatnonzero(ok)
                # blocks covered by the hit k-windows (for the gated spectrum): all blocks i..i+kadj-1
                cov = np.unique(np.concatenate([hit + j for j in range(kadj)]))
                passing.append(dict(ci=ci, bw=bw, fc=fc, sf=sf, dir=name, R=float(mk[k]), score=float(mk[k] / T),
                                    n_hits=int(len(hit)), n_blocks=int(len(main)), block=k, start=int(lb['start'][k]),
                                    span=int(lb['span']), hit_idx=cov, main=main, lb=lb,
                                    adjacent=True))''')
open('analyze.py', 'w').write(a)

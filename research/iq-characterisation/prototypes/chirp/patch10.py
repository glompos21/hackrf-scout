g = open('gated.py').read()
g = g.replace('''    bridged = np.convolve(mask.astype(np.int8), np.ones(7, np.int8), mode='same') >= 1
    bridged = np.convolve(bridged.astype(np.int8), np.ones(7, np.int8), mode='same') >= 7        # close, not dilate
    mask = mask | bridged & (np.convolve(mask.astype(np.int8), np.ones(7, np.int8), mode='same') >= 2)
''', '''    dil = np.convolve(mask.astype(np.int8), np.ones(7, np.int8), mode='same') >= 1             # closing: dilate ...
    mask = mask | (np.convolve(dil.astype(np.int8), np.ones(7, np.int8), mode='same') >= 7)    # ... then erode
''')
open('gated.py', 'w').write(g)

a = open('analyze.py').read()
# insert refine_sf helper before chirp_stage
helper = '''def refine_sf(src, mean, rate, cand, bw, centre, cfg, T):
    """Matched re-analysis of the segments where the chirp test fired: channel centred on the gated-PSD centre with the
    bandwidth hint, every SF, the strongest line decides the SF (a mismatched channel shows lines at shifted
    positions, so the SF read from it can be off by 1-2).  Returns dict(sf, R, second, by_sf)."""
    D = int(round(rate / bw))
    tile = D * 4096
    lb0 = cand['lb']
    hit = cand['hit_idx']
    Dc = int(round(rate / cand['bw']))
    spans = []
    for h in hit[:cfg.get('refine_blocks', 24)]:
        a = int(lb0['start'][h]) * Dc
        spans.append((max(0, a - lb0['span'] * Dc // 2), a + lb0['span'] * Dc + lb0['span'] * Dc // 2))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    best = {}
    ch = fe.Channeliser(rate, centre, bw, tile)
    mean_c = np.complex64(mean)
    used = 0
    for a, b in merged:
        if used > cfg.get('refine_max', 8_000_000):
            break
        ntile = max(1, int(math.ceil((b - a) / tile)))
        x = src.read(a, ntile * tile)
        if len(x) < tile:
            continue
        x = x[:len(x) // tile * tile] - mean_c
        y = np.concatenate([ch.block(x[t * tile:(t + 1) * tile]) for t in range(len(x) // tile)])
        used += len(x)
        z = css.lag_z(y)
        for sf in SFS:
            lb = css.lag_blocks(y, sf, L=cfg['L'], z=z)
            if lb is None:
                continue
            main = lb['up'] if cand['dir'] == 'up' else lb['dn']
            opp = lb['dn'] if cand['dir'] == 'up' else lb['up']
            ok = main > cfg['asym'] * opp
            v = float(np.max(np.where(ok, main, 0))) if ok.any() else 0.0
            best[sf] = max(best.get(sf, 0.0), v)
    if not best:
        return None
    order = sorted(best.items(), key=lambda kv: -kv[1])
    return dict(sf=order[0][0], R=order[0][1], second=order[1][1] if len(order) > 1 else 0.0,
                by_sf={int(k): float(v) for k, v in best.items()})


'''
a = a.replace('def chirp_stage(', helper + 'def chirp_stage(', 1)
# replace the SF mapping logic
old_start = a.index("            mu = cand['bw'] ** 2 / (1 << cand['sf'])")
old_end = a.index("            cand['bw_final'] = bw_final")
a = a[:old_start] + a[old_end:]
a = a.replace("            cand['sf_final'] = int(round(sf_f))\n            cand['centre'] = band['centre']\n            cand['band_confirmed'] = True", "            cand['centre'] = band['centre']\n            cand['band_confirmed'] = True\n            rf = refine_sf(src, p1['mean'], rate, cand, bw_final, band['centre'], cfg, T)\n            cand['refine'] = rf\n            cand['sf_final'] = rf['sf'] if rf and rf['R'] > T else cand['sf'] if bw_final == cand['bw'] else None")
a = a.replace("            cand['sf_final'] = cand['sf']\n            cand['centre'] = cand['fc']", "            cand['sf_final'] = cand['sf']\n            cand['centre'] = cand['fc']")
a = a.replace("    out['flag'] = True\n    out['best'] = {k: v for k, v in final.items() if k not in ('hit_idx', 'ci', 'main', 'lb')}", "    out['flag'] = True\n    out['best'] = {k: v for k, v in final.items() if k not in ('hit_idx', 'ci', 'main', 'lb')}\n    if final['sf_final'] is None:\n        final['sf_final'] = final['sf']")
open('analyze.py', 'w').write(a)

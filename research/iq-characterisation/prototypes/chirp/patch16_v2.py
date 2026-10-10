"""v2 fix (applied AFTER the frozen evaluation finished): the matched refinement must choose the chirp direction
itself.  A candidate found in a mismatched (over-sampled) channel can show its line on the opposite sign (the
sign-flip modulation at every frequency wrap moves the line to f0 -/+ 1/(2T)), so refining with the candidate's
direction gave R = 0 for SF12 in 17 of 1,400 frozen-eval detections (they were reported 'low' confidence)."""
a = open('analyze.py').read()
a = a.replace("""            main = lb['up'] if cand['dir'] == 'up' else lb['dn']
            opp = lb['dn'] if cand['dir'] == 'up' else lb['up']
            ok = main > cfg['asym'] * opp
            v = float(np.max(np.where(ok, main, 0))) if ok.any() else 0.0
            best[sf] = max(best.get(sf, 0.0), v)""", """            for dname, main, opp in (('up', lb['up'], lb['dn']), ('down', lb['dn'], lb['up'])):
                ok = main > cfg['asym'] * opp
                v = float(np.max(np.where(ok, main, 0))) if ok.any() else 0.0
                if v > bestd.get((dname, sf), 0.0):
                    bestd[(dname, sf)] = v
                best[sf] = max(best.get(sf, 0.0), v)""")
a = a.replace("    best = {}\n    ch = fe.Channeliser(rate, centre, bw, tile)", "    best = {}\n    bestd = {}\n    ch = fe.Channeliser(rate, centre, bw, tile)")
a = a.replace("""    out = dict(sf=int(srt[0][0]), R=float(srt[0][1]), second=float(srt[1][1]) if len(srt) > 1 else 0.0,
               by_sf={int(k): float(v) for k, v in best.items()}, n_seg=len(ys))""", """    kd = max(bestd.items(), key=lambda kv: kv[1])
    out = dict(sf=int(srt[0][0]), R=float(srt[0][1]), second=float(srt[1][1]) if len(srt) > 1 else 0.0,
               by_sf={int(k): float(v) for k, v in best.items()}, n_seg=len(ys), dir=kd[0][0])
    dirn = out['dir']""")
a = a.replace("R = dc['up'] if cand['dir'] == 'up' else dc['dn']\n            hits += int((R > thr).sum())", "R = dc['up'] if dirn == 'up' else dc['dn']\n            hits += int((R > thr).sum())")
a = a.replace("run = max(run, css.preamble_run(dc if cand['dir'] == 'up' else dict(up=dc['dn'], bin_up=dc['bin_dn']), thr, M))", "run = max(run, css.preamble_run(dc if dirn == 'up' else dict(up=dc['dn'], bin_up=dc['bin_dn']), thr, M))")
open('analyze.py', 'w').write(a)
print('patched')

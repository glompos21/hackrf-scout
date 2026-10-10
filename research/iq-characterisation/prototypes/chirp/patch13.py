a = open('analyze.py').read()
a = a.replace('''    out = dict(T=T, n_trials=n_trials, p_block=p_block, max_R=mx, max_score=mx / T, n_pass=len(passing))''', '''    out = dict(T=T, n_trials=n_trials, p_block=p_block, max_R=mx, max_score=mx / T, n_pass=len(passing))
    # diagnostic: the best score (max over tests of min-R-over-k-adjacent-blocks / threshold_k) for k = 1, 2, 3
    sc = {}
    for kk in (1, 2, 3):
        Tk = lag_null_threshold(L, p_block, kk)
        best_k = 0.0
        for ci, bw, fc, sf, lb in lag:
            for main, opp in ((lb['up'], lb['dn']), (lb['dn'], lb['up'])):
                mk = kmin(np.where(main > cfg['asym'] * opp, main, 0.0), kk)
                if len(mk):
                    best_k = max(best_k, float(mk.max()))
        sc[kk] = best_k / Tk
    out['k_scores'] = sc''')
open('analyze.py', 'w').write(a)

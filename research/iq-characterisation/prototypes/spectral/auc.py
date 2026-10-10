"""Single-cue separability (ROC AUC, rank statistic) on the evaluation features.  A cue that is missing (None) on a
capture counts as 'no evidence' (ranked below everything) and the coverage is printed."""
import sys, json
sys.path.insert(0, '.')
import numpy as np, summ, harness, truth_class, rules

def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    allv = np.concatenate([pos, neg]); 
    order = np.argsort(allv, kind='mergesort'); ranks = np.empty(len(allv)); 
    # average ranks for ties
    sv = allv[order]; i = 0; r = np.empty(len(allv))
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def get(rows, f, sign=1.0, missing=-1e9):
    out = []; miss = 0
    for r in rows:
        v = f(r)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out.append(missing); miss += 1
        else: out.append(sign * float(v))
    return np.array(out), 1 - miss / max(1, len(rows))

def comb_gated_A(r):
    F = r['feat']; A = F.get('cp_A'); c = F.get('cp_comb'); z = F.get('cp_z')
    if A is None or z is None or z < 10: return 0.0
    return A * (1.0 if (c is None or c <= 0.5) else 0.0)

CUES = {
 'noise vs any signal': dict(pos=lambda r: r['kind'] != 'noise', neg=lambda r: r['kind'] == 'noise', cues={
     'struct_z': (lambda r: r['feat'].get('struct_z'), 1), 'occ_frac': (lambda r: r['feat'].get('occ_frac'), 1),
     'n_lines>0': (lambda r: float(r['feat'].get('n_lines', 0) > 0), 1)}),
 'wide vs narrow (by nominal B)': dict(pos=lambda r: r['true'] == 'wide' and r['care'], neg=lambda r: r['true'] == 'narrow' and r['care'], cues={
     'obw99_hz': (lambda r: r['feat'].get('obw99_hz'), 1), 'bw_3db_hz': (lambda r: r['feat'].get('bw_3db_hz'), 1),
     'bw_20db_hz': (lambda r: r['feat'].get('bw_20db_hz'), 1), 'occ_frac': (lambda r: r['feat'].get('occ_frac'), 1)}),
 'OFDM vs other wide/medium signals': dict(pos=lambda r: r['kind'] == 'ofdm', neg=lambda r: r['true'] in ('medium', 'wide') and r['kind'] not in ('ofdm', 'hopper'), cues={
     'cp_z': (lambda r: r['feat'].get('cp_z'), 1), 'cp_A': (lambda r: r['feat'].get('cp_A'), 1), 'cp_A gated by comb': (comb_gated_A, 1),
     'flat_db (flatter=higher)': (lambda r: r['feat'].get('flat_db'), 1), 'ptm_db (low=OFDM)': (lambda r: r['feat'].get('ptm_db'), -1),
     'edge_ratio (low=OFDM)': (lambda r: r['feat'].get('edge_ratio'), -1), 'obw99_hz': (lambda r: r['feat'].get('obw99_hz'), 1)}),
 'hopper vs all other signals': dict(pos=lambda r: r['kind'] == 'hopper', neg=lambda r: r['kind'] not in ('hopper', 'noise'), cues={
     'hop_n_ch': (lambda r: r['feat'].get('hop_n_ch'), 1), 'hop_n_hops': (lambda r: r['feat'].get('hop_n_hops'), 1),
     'n_lobes': (lambda r: r['feat'].get('n_lobes'), 1), 'n_clusters': (lambda r: r['feat'].get('n_clusters'), 1),
     'obw99_hz': (lambda r: r['feat'].get('obw99_hz'), 1)}),
 'CW vs other narrow signals': dict(pos=lambda r: r['kind'] == 'cw', neg=lambda r: r['true'] == 'narrow' and r['care'], cues={
     'obw_bins (low=CW)': (lambda r: r['feat'].get('obw_bins'), -1), 'n_lines==1': (lambda r: float(r['feat'].get('n_lines') == 1), 1),
     'ptm_db': (lambda r: r['feat'].get('ptm_db'), 1), 'edge_ratio': (lambda r: r['feat'].get('edge_ratio'), 1)}),
}

if __name__ == '__main__':
    T = rules.load_thr()
    import eval_rules
    rows = eval_rules.annotate(summ.load(sys.argv[1]), T)
    res = {}
    for cond in ('clean', 'imp'):
        rc = [r for r in rows if r['cond'] == cond]
        for q, d in CUES.items():
            print('\n== %s   [%s]   AUC by SNR (coverage = share of positives with a value)' % (q, cond))
            print('%-26s' % 'cue' + ''.join('%14d' % s for s in (0, 5, 10, 15, 20, 30)))
            for name, (f, sg) in d['cues'].items():
                line = '%-26s' % name
                for s in (0, 5, 10, 15, 20, 30):
                    sub = [r for r in rc if r['snr'] == s or r['kind'] == 'noise']
                    P = [r for r in sub if d['pos'](r)]; N = [r for r in sub if d['neg'](r)]
                    pv, cov = get(P, f, sg); nv, _ = get(N, f, sg)
                    a = auc(pv, nv)
                    line += '  %5.3f (%3.0f%%)' % (a, 100 * cov)
                    res.setdefault(q, {}).setdefault(cond, {}).setdefault(name, {})[s] = [a, cov]
                print(line)
    json.dump(res, open('results/auc.json', 'w'), indent=1)

"""False-alarm campaign on pure noise (seeds 100-299, never used for tuning)."""
import sys, json
sys.path.insert(0, '.')
import numpy as np, harness, gen, sp_stream, sp_feat, rules
from multiprocessing import Pool
T = rules.load_thr()

def one(a):
    rate, secs, seed, dc = a
    r = gen.make('noise', rate, secs, 0.0, seed=seed, dc=dc, return_ci8=False)
    sp = sp_stream.analyse_stream(lambda: sp_stream.iter_array(r['iq']), rate)
    F = sp_feat.features(sp)
    lab, conf, why = rules.classify(F, T)
    return dict(rate=rate, secs=secs, seed=seed, dc=dc, label=lab, struct_z=F['struct_z'], n_lines=F['n_lines'],
                detected=F['detected'], cp_z=F.get('cp_z'), hop_n_ch=F.get('hop_n_ch'), struct_scale=F.get('struct_scale'),
                dc_resid_z=F.get('dc_resid_z'))

if __name__ == '__main__':
    tasks = []
    for seed in range(100, 300):
        for dc in (0.0, 6.0):
            tasks.append((2e6, 1.0, seed, dc))
            tasks.append((10e6, 0.5, seed, dc))
    for seed in range(100, 140):
        tasks.append((2e6, 5.0, seed, 6.0))
    with Pool(4) as pool:
        res = pool.map(one, tasks, chunksize=4)
    json.dump(res, open('results/noise_fpr.json', 'w'))
    import collections
    for key in sorted(set((r['rate'], r['secs'], r['dc']) for r in res)):
        rr = [r for r in res if (r['rate'], r['secs'], r['dc']) == key]
        z = np.array([r['struct_z'] for r in rr])
        print('rate %.0e secs %.1f dc %.0f: n=%d  labels %s  struct_z max %.2f  p99 %.2f  lines>0: %d' % (key[0], key[1], key[2], len(rr), dict(collections.Counter(r['label'] for r in rr)), z.max(), np.percentile(z, 99), sum(r['n_lines'] > 0 for r in rr)))

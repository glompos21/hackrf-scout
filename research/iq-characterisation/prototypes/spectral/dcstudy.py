"""DC spike / CFO / carrier-at-DC studies (reported numbers, eval-type seeds 100-109)."""
import sys, json, time
sys.path.insert(0,'..'); sys.path.insert(0,'.')
import numpy as np, gen, harness, sp_stream, sp_feat, realism
from multiprocessing import Pool

def one(args):
    case, snr, seed, extra, remove_dc, wander, tag = args
    kind, rate, secs, kw = harness.CASES[case]
    p = harness.cond_params(case, 'clean', seed, 2)
    p.update(extra)
    if kind == 'noise': p.pop('on_fraction', None)
    r = gen.make(kind, rate, secs, float(snr), seed=seed, return_ci8=False, **p, **kw)
    iq = r['iq']
    if wander: iq = realism.apply(iq, rate, seed=seed, dc_wander_lsb=wander, dc_corner_hz=500.0)
    sp = sp_stream.analyse_stream(lambda: sp_stream.iter_array(iq), rate, remove_dc=remove_dc)
    F = sp_feat.features(sp)
    keep = {k: F.get(k) for k in ('struct_z','detected','n_lines','obw99_hz','snr_est_db','flat_db','dc_resid_z','dc_rel_db','cp_z','lines','obw_centre_hz','n_lobes','floor_ok')}
    return dict(case=case, snr=snr, seed=seed, tag=tag, extra=extra, truth_center=r['truth']['center_hz'], truth_bw=r['truth']['occupied_bw_hz'], f=keep)

if __name__ == '__main__':
    tasks = []
    seeds = range(100, 110)
    modes = [  # tag, extra, remove_dc, wander
        ('dc0',            dict(),                True,  0.0),
        ('dc6_raw',        dict(dc=6.0),          False, 0.0),
        ('dc6_mean',       dict(dc=6.0),          True,  0.0),
        ('dc12_raw',       dict(dc=12.0),         False, 0.0),
        ('dc12_mean',      dict(dc=12.0),         True,  0.0),
        ('dc0_wander0.5',  dict(),                True,  0.5),
        ('dc6_wander0.5',  dict(dc=6.0),          True,  0.5),
        ('dc6_wander2',    dict(dc=6.0),          True,  2.0),
        ('cfo4k',          dict(cfo_hz=4e3),      True,  0.0),
        ('dc6_cfo4k',      dict(dc=6.0, cfo_hz=4e3), True, 0.0),
    ]
    for case in ('noise2','cw','nfm-voice','qpsk','lora-sf7','ofdm10','hopper10'):
        for tag, extra, rd, wd in modes:
            for s in seeds:
                tasks.append((case, 10, s, extra, rd, wd, tag))
    # carrier at / near DC
    for case in ('cw','am-tone','nfm-tone'):
        for off in (0.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0):
            for dc in (0.0, 6.0):
                for s in range(100, 106):
                    kind, rate, secs, kw = harness.CASES[case]
                    tasks.append((case, 10, s, dict(offset_hz=off, dc=dc), True, 0.0, 'atdc_off%g_dc%g' % (off, dc)))
    t = time.time()
    with Pool(4) as pool:
        res = pool.map(one, tasks, chunksize=4)
    json.dump(res, open('results/dcstudy.json', 'w'), default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print('done', len(res), time.time() - t)

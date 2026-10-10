import sys
sys.path.insert(0, '.')
import numpy as np
from tab import *
path = sys.argv[1]; scen = sys.argv[2].split(',') if len(sys.argv) > 2 else None
rows = load(path)
names = scen or sorted({r['scen'] for r in rows})
KEYS = [('t_duty','tD'),('duty','D'),('ev_iou','iou'),('all_m4','m4all'),('on_m4','m4on'),('on_m4_sig','sig'),('on_snr_ch_db','snrch'),('o_on_m4_sig','osig'),('line_over_thr_db','lineT'),('det_any','any'),('ch_bw','bw'),('run_n_runs','nrun')]
for nm in names:
    for om in ('native', 'gated'):
        rs = sel(rows, scen=nm, onmode=om)
        if not rs: continue
        snrs = sorted({r['snr'] for r in rs})
        print('== %s [%s] n=%d' % (nm, om, len(rs)))
        print('  snr  ' + ' '.join('%7s' % k[1] for k in KEYS))
        for s in snrs:
            rr = sel(rs, snr=s)
            vals = []
            for k, _ in KEYS:
                v = col(rr, k)
                vals.append('%7.2f' % med(v) if k not in ('ch_bw','run_n_runs') else '%7.0f' % med(v))
            print('  %3d  ' % s + ' '.join(vals))

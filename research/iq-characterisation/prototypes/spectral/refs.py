"""Reference bandwidths: nominal B (truth), the independent measured 99 % bw (meas.containment) on a clean
30 dB capture, and the 'hull of bands' for multi-channel signals.  Seeds 900-903 (not tune, not eval)."""
import sys, json
sys.path.insert(0,'..'); sys.path.insert(0,'.')
import numpy as np, gen, meas, harness
out={}
for case,(kind,rate,secs,kw) in harness.CASES.items():
    if kind=='noise': continue
    b99=[];b999=[];nom=None;hull=None
    for seed in (900,901,902,903):
        p=harness.cond_params(case,'clean',seed,5)
        r=gen.make(kind,rate,secs,30.0,seed=seed,return_ci8=False,**p,**kw)
        tr=r['truth']
        c=meas.containment(r['iq'],tr)
        b99.append(c['bw99']); b999.append(c['bw999']); nom=tr['occupied_bw_hz']
        hull=max(b[1] for b in tr['bands_hz'])-min(b[0] for b in tr['bands_hz'])
    out[case]=dict(nominal=nom,bw99=float(np.median(b99)),bw999=float(np.median(b999)),hull=hull)
    print('%-15s nominal %10.0f  meas bw99 %10.0f  bw99.9 %10.0f hull %10.0f'%(case,nom,out[case]['bw99'],out[case]['bw999'],hull))
json.dump(out,open('results/refs.json','w'),indent=1)

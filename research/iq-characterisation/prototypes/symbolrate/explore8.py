import sys
import numpy as np
sys.path.insert(0, '..')
import gen
from explore1 import prep
from explore7 import ook_runs, estimate_T
kind=sys.argv[1]; snrs=[float(s) for s in sys.argv[2].split(',')]
kw=eval(sys.argv[3]) if len(sys.argv)>3 else {}
for snr in snrs:
    r = gen.make(kind, 2e6, 4.0, snr, seed=1, offset_hz=60e3, cfo_hz=3e3, dc=4.0, return_ci8=False, **kw)
    tr = r['truth']; T0=tr['symbol_rate']
    rec, fs, n_ch, info, b, fb = prep(r, 2e6)
    line=[]
    for Kc in (1.5,3,5,8,12):
        out = ook_runs(rec, fs, n_ch, info['neb'], Kc=Kc)
        if out is None: line.append('Kc%.1f none'%Kc); continue
        est = estimate_T(out[0], fs, info['neb'], tmin=Kc/info['neb'])
        line.append('Kc%.1f: %s'%(Kc, 'None' if est is None else '%.0f Hz sup %.2f'%(1/est[0], est[3])))
    print('snr',snr,'truth',T0,'|',' | '.join(line))

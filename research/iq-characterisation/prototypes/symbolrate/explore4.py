import sys, time
import numpy as np
sys.path.insert(0, '..')
import gen, frontend as fe
from common import *
from explore1 import prep
kinds = sys.argv[1].split(','); snrs=[float(s) for s in sys.argv[2].split(',')]
for kind in kinds:
    rate = 2e6 if kind!='gfsk' else 4e6
    for snr in snrs:
        r = gen.make(kind, rate, 4.0, snr, seed=1, offset_hz=60e3, cfo_hz=3e3, dc=4.0, return_ci8=False)
        tr = r['truth']
        rec, fs, n_ch, info, b, fb = prep(r, rate)
        p = (np.abs(rec) ** 2)
        t=time.time()
        nfft = 16384
        f,S,ns = welch_real(p.astype(np.float32), fs, nfft)
        # search 
        res = line_stat(f,S,ns, 200, 0.45*fs)
        # x^2 and x^4 lines (complex spectrum)
        out=[]
        for q in (2,4):
            xq = rec**q
            fq,Sq,nq = welch_real(xq.astype(np.complex64), fs, nfft)
            lq = line_stat(fq,Sq,nq,-0.5*fs,0.5*fs)
            out.append((q, round(lq['f'],0), round(lq['ratio'],1)))
        print('%-5s snr %3.0f truth Rs %s | |y|^2 line f=%.0f ratio=%.1f z=%.0f | x^q lines %s | %.2fs' % (kind,snr,tr['symbol_rate'],res['f'],res['ratio'],res['z'],out,time.time()-t))

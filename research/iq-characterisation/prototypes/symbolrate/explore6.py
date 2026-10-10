import sys, time
import numpy as np
sys.path.insert(0, '..')
import gen, frontend as fe
from common import *
from explore1 import prep
kind=sys.argv[1]; snrs=[float(s) for s in sys.argv[2].split(',')]
kw=eval(sys.argv[3]) if len(sys.argv)>3 else {}
rate=float(sys.argv[4]) if len(sys.argv)>4 else 2e6
def at_truth(f,S,f0):
    k=np.argmin(np.abs(f-f0)); w=slice(max(0,k-2),k+3)
    pk=S[w].max()
    ring=np.concatenate([S[max(0,k-30):k-4],S[k+5:k+30]])
    return pk/np.median(ring)
def best_line(f,S,lo,hi):
    sel=np.flatnonzero((f>=lo)&(f<=hi))
    # ratio to running median of 60 bins
    from scipy.ndimage import median_filter
    return None
for snr in snrs:
    r = gen.make(kind, rate, 4.0, snr, seed=1, offset_hz=60e3, cfo_hz=3e3, dc=4.0, return_ci8=False, **kw)
    tr = r['truth']; T0=tr['symbol_rate']
    rec, fs, n_ch, info, b, fb = prep(r, rate)
    out=[]
    # dev estimate from smoothed disc
    z = rec[1:]*np.conj(rec[:-1])
    zs = movsum_complex(z, max(1,int(fs/(2*T0))//2))
    s = np.angle(zs)*fs/(2*np.pi); s=s-np.median(s)
    dev = np.mean(np.abs(s))
    # (A) delay and multiply real part
    for mult in (0.5,1.0,2.0):
        tau=max(1,int(round(mult*fs/(4*dev))))
        c=(rec[tau:]*np.conj(rec[:-tau]))
        u=np.real(c)/ (np.abs(c).mean()) 
        f,S,ns=welch_real(u.astype(np.float32),fs,8192)
        out.append('DM tau=%d(%.1fus): %.1f'%(tau,tau/fs*1e6,at_truth(f,S,T0)))
    # (C) hysteresis binarization -> edge impulse train
    for Lq in (1,4):
        zs = movsum_complex(z, Lq)
        s = np.angle(zs)*fs/(2*np.pi); s=s-np.median(s)
        bsig=(s>0).astype(np.float32)
        e=np.abs(np.diff(bsig))
        f,S,ns=welch_real(e,fs,8192)
        out.append('edge L%d: %.1f'%(Lq,at_truth(f,S,T0)))
    print('snr %3.0f fs %.0f dev~%.0f | %s'%(snr,fs,dev,' | '.join(out)))

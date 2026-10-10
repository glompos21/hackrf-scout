import sys, time
import numpy as np
sys.path.insert(0, '..')
import gen, frontend as fe
from common import *
from explore1 import prep
kind=sys.argv[1]; snrs=[float(s) for s in sys.argv[2].split(',')]
kw=eval(sys.argv[3]) if len(sys.argv)>3 else {}
rate=float(sys.argv[4]) if len(sys.argv)>4 else 2e6
def at_truth(f,S,ns,f0,floor_lo,floor_hi):
    k=np.argmin(np.abs(f-f0)); w=slice(max(0,k-2),k+3)
    pk=S[w].max()
    ring=np.concatenate([S[max(0,k-30):k-4],S[k+5:k+30]])
    # global: strongest other bin ratio in search band
    sel=(f>=floor_lo)&(f<=floor_hi)
    fl=np.median(S[sel])
    return pk/np.median(ring), pk/fl, f[w][np.argmax(S[w])]
for snr in snrs:
    r = gen.make(kind, rate, 4.0, snr, seed=1, offset_hz=60e3, cfo_hz=3e3, dc=4.0, return_ci8=False, **kw)
    tr = r['truth']; T0=tr['symbol_rate']
    rec, fs, n_ch, info, b, fb = prep(r, rate)
    z = rec[1:]*np.conj(rec[:-1])
    p=np.abs(rec)**2
    out=[]
    for Ls in (1,2,4):
        zs = movsum_complex(z, Ls)
        s = np.angle(zs)*fs/(2*np.pi)
        w = np.abs(zs)  # weight
        s = s - np.median(s)
        series={'f^2':s**2, 'abs f':np.abs(s), 'df^2':(s[2:]-s[:-2])**2}
        for name,u in series.items():
            nfft=8192
            f,S,ns=welch_real(u.astype(np.float32),fs,nfft)
            ra,rb,fp=at_truth(f,S,ns,T0,100,0.4*fs)
            out.append('%s/L%d: %.1f(%.0f)'%(name,Ls,ra,fp))
    f,S,ns=welch_real(p.astype(np.float32),fs,8192)
    ra,rb,fp=at_truth(f,S,ns,T0,100,0.4*fs)
    out.append('|y|^2: %.1f'%ra)
    print('snr %3.0f fs %.0f band %.0f | %s'%(snr,fs,b['hi']-b['lo'],' | '.join(out)))

import sys, time
import numpy as np
sys.path.insert(0, '..')
import gen, frontend as fe
from common import *
from explore1 import prep

def ook_runs(rec, fs, n_ch, neb, Kc=3.0, verbose=False):
    p = np.abs(rec).astype(np.float64)**2
    Lc = max(1, int(round(Kc*fs/neb)))
    a = movmean(p, Lc)
    # frame mask by long window
    K=24; Lp=int(np.ceil(K*fs/neb)); af = movmean(p, Lp)
    fm = af > n_ch*(1+5/np.sqrt(K))
    # dilate frame mask by Lp
    fmd = movmean(fm.astype(float), 2*Lp) > 0
    if fmd.sum() < 10*Lp:
        return None
    ax = a[fmd]
    # on level: 90th percentile of smoothed power within frames
    P_hi = np.percentile(ax, 90)
    lo_l = n_ch
    if P_hi < 1.5*n_ch: return None
    th_h = n_ch + 0.6*(P_hi-n_ch); th_l = n_ch+0.3*(P_hi-n_ch)
    # schmitt
    st = np.full(len(a), -1, np.int8)
    st[a>th_h]=1; st[a<th_l]=0
    idx = np.where(st>=0, np.arange(len(a)), 0)
    idx = np.maximum.accumulate(idx)
    s = st[idx]; s[s<0]=0
    s = (s>0)
    # only inside dilated frames
    st_,ln,val = runs(s)
    # edge positions
    edges = st_[1:]
    d = np.diff(edges)/fs     # run durations between consecutive edges
    # keep runs where both bounding edges are inside frames
    ins = fmd[edges[:-1]] & fmd[edges[1:]]
    d = d[ins]
    return d, P_hi

def estimate_T(d, fs, neb, tmin=None):
    tmin = tmin or 2.0/neb
    d = d[(d>=tmin)]
    if len(d)<20: return None
    lg = np.log10(d)
    h,e = np.histogram(lg, bins=np.arange(lg.min()-0.05, lg.max()+0.05, 1/40.))
    hs = np.convolve(h, np.array([1,2,3,2,1])/9., mode='same')
    # first peak with mass
    tot = hs.sum()
    for i in range(1,len(hs)-1):
        if hs[i]>=hs[i-1] and hs[i]>hs[i+1] and hs[i]>0.04*hs.max():
            c = 10**((e[i]+e[i+1])/2)
            break
    else:
        return None
    # refine by integer grid LS
    T=c
    for it in range(5):
        sel = (d>0.6*T)&(d<12*T)
        n = np.maximum(np.round(d[sel]/T),1)
        T = (n*d[sel]).sum()/(n*n).sum()
    sel = (d>0.6*T)&(d<12*T)
    n = np.maximum(np.round(d[sel]/T),1)
    res = (d[sel]-n*T)/T
    return T, c, np.sqrt((res**2).mean()), (np.abs(res)<0.2).mean(), sel.sum(), len(d)

if __name__=='__main__':
    kind=sys.argv[1]; snrs=[float(s) for s in sys.argv[2].split(',')]
    kw=eval(sys.argv[3]) if len(sys.argv)>3 else {}
    for snr in snrs:
        r = gen.make(kind, 2e6, 4.0, snr, seed=1, offset_hz=60e3, cfo_hz=3e3, dc=4.0, return_ci8=False, **kw)
        tr = r['truth']; T0=tr['symbol_rate']
        rec, fs, n_ch, info, b, fb = prep(r, 2e6)
        out = ook_runs(rec, fs, n_ch, info['neb'])
        if out is None:
            print('snr',snr,'no frames'); continue
        d,P=out
        est = estimate_T(d, fs, info['neb'])
        print('snr %3.0f fs %.0f ndur %d | truth T=%.1fus  est %s'%(snr, fs, len(d), 1e6/T0, None if est is None else 'T=%.1fus (cluster %.1f) rms %.3f support %.2f n=%d/%d -> baud %.0f'%(est[0]*1e6,est[1]*1e6,est[2],est[3],est[4],est[5],1/est[0])))

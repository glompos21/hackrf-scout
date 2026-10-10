import sys
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'..')
import scen, frontend as fe, features
from features import *
res_all=[]
for v in ('ook_pwm','ook_man','ook_nrz'):
  for snr in (10,15,20,30):
    for seed in range(3):
        res,imp=scen.make(v,snr,seed,'cont')
        rate=res['truth']['rate']
        src=fe.ArraySource(res['iq'],rate)
        sc=fe.scan_psd(src); fb=fe.find_bands(sc,rate); b=fb['bands'][0]
        bw=1.3*(b['hi']-b['lo'])+2e3; D=fe.choose_decimation(rate,bw)
        rec,fs,info=fe.channelize_select(src,b['centroid'],bw,D,fb['n0'],dc=sc['dc'])
        neb=info['neb']; n_ch=fb['n0']*neb
        p=(np.abs(rec)**2).astype(np.float64)
        fmask,Lp=frame_mask(p.astype(np.float32),fs,neb,n_ch)
        Kc=6.0; Lc=max(1,int(round(Kc*fs/neb))); a=movmean(p,Lc)
        fmd=movmean(fmask.astype(float),2*Lp)>0; ax=a[fmd]
        thr_o,sep=otsu(ax[::max(1,len(ax)//200000)],128); P_hi=float(np.median(ax[ax>=thr_o]))
        th_h=n_ch+0.6*(P_hi-n_ch); th_l=n_ch+0.3*(P_hi-n_ch)
        s=schmitt(a,th_l,th_h); st,ln,val=runs(s); edges=st[1:]
        # edge times with sub-sample interpolation on the 50% crossing between neighbours
        d=np.diff(edges)/fs; ins=fmd[edges[:-1]]&fmd[edges[1:]]; d=d[ins]
        est=estimate_unit(d,tmin=Kc/neb)
        if not est: continue
        T0=est['T']; sr=res['truth']['symbol_rate']
        # same-polarity intervals
        rising=edges[val[1:]]  # edges where new state is True
        falling=edges[~val[1:]]
        def lsT(dd, T):
            for _ in range(5):
                n=np.maximum(np.round(dd/T),1); T=(n*dd).sum()/(n*n).sum()
            return T
        drr=np.diff(rising)/fs; dff=np.diff(falling)/fs
        okr=drr[(drr<12*T0)&(drr>0.6*T0)]; okf=dff[(dff<12*T0)&(dff>0.6*T0)]
        both=np.concatenate([okr,okf])
        T1=lsT(both,T0)
        print('%-8s %2d s%d truth %.1f  comb %.1f (%.2f%%)  samepol %.1f (%.2f%%)  nrr %d'%(v,snr,seed,sr,1/est['T'],100*(1/est['T']/sr-1),1/T1,100*(1/T1/sr-1),len(both)))

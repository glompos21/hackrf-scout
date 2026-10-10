import sys; sys.path.insert(0,'..'); sys.path.insert(0,'.')
import numpy as np, gen, harness, sp_stream as ss, sp_feat as sf
for case in ('fsk2-gauss','ofdm10'):
    kind,rate,secs,kw=harness.CASES[case]
    p=harness.cond_params(case,'clean',3,3)
    r=gen.make(kind,rate,secs,15.0,seed=3,return_ci8=False,**p,**kw); tr=r['truth']
    sp=ss.analyse_stream(lambda: ss.iter_array(r['iq']),rate); F=sf.features(sp)
    f,P=sp['f'],sp['P']
    print(case,'obw',F['obw99_hz'],'lo',F['obw_lo_hz'],'hi',F['obw_hi_hz'],'lobes',F['n_lobes'],'flat',F['flat_db'],'lines',F['lines'])
    sel=(f>=F['obw_lo_hz'])&(f<=F['obw_hi_hz'])
    g=max(1,int(round(F['obw99_hz']/(f[1]-f[0])/32)))
    Pq=sf.rebin(P[sel],g); print(np.round(10*np.log10(Pq/Pq.max()),1))

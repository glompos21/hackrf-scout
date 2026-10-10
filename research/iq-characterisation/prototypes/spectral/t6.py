import sys; sys.path.insert(0,'..'); sys.path.insert(0,'.')
import numpy as np, gen, harness, sp_stream as ss, sp_feat as sf, meas
for case in ('fsk2-gauss','nfm-voice','am-voice'):
    kind,rate,secs,kw=harness.CASES[case]
    p=harness.cond_params(case,'clean',3,5)
    r=gen.make(kind,rate,secs,30.0,seed=3,return_ci8=False,**p,**kw); tr=r['truth']
    sp=ss.analyse_stream(lambda: ss.iter_array(r['iq']),rate); F=sf.features(sp)
    f,P=sp['f'],sp['P']; n0=tr['noise_psd_lsb2_per_hz']; df=f[1]-f[0]
    print(case,'n0 est %.3g true %.3g'%(F['n0'],n0),'obw',F['obw99_hz'],'lo/hi',F['obw_lo_hz'],F['obw_hi_hz'],'center true',tr['center_hz'])
    S=np.maximum(P-n0,0); # exact floor, no threshold
    m=(np.abs(f-tr['center_hz'])<tr['occupied_bw_hz']*2)
    c=np.cumsum(S[m])/S[m].sum(); ff=f[m]
    for pp in (0.005,0.01,0.025,0.975,0.99,0.995): print('  q%.3f at %.0f'%(pp,ff[np.searchsorted(c,pp)]),end='')
    print('\n  exact-floor 99%% bw = %.0f'%(ff[np.searchsorted(c,0.995)]-ff[np.searchsorted(c,0.005)]))
    # what fraction of the power lies outside +-bw/2 around centre in the hard-thresholded estimate

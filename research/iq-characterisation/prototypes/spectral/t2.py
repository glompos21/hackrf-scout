import sys, time, numpy as np
sys.path.insert(0,'..'); sys.path.insert(0,'.')
import gen, meas, sp_stream as ss, sp_feat as sf
r=gen.make('ofdm',10e6,0.5,15,seed=0,return_ci8=False); tr=r['truth']
print(tr['bands_hz'], tr['occupied_bw_hz'])
sp=ss.analyse_stream(lambda: ss.iter_array(r['iq']),10e6)
f,P=sp['f'],sp['P']
db=10*np.log10(rebin:=P[:len(P)//64*64].reshape(-1,64).mean(1)); ff=f[:len(P)//64*64].reshape(-1,64).mean(1)
for a,b in zip(ff[::8],db[::8]): print('%8.0f %6.1f'%(a,b))
F=sf.features(sp); print({k:F[k] for k in ('n0','floor_frac','floor_ok','floor_g','floor_nsel')})

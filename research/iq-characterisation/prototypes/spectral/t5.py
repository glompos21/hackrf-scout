import sys; sys.path.insert(0,'..'); sys.path.insert(0,'.')
import numpy as np, gen, realism, sp_stream as ss, sp_feat as sf
r=gen.make('noise',2e6,0.5,0,seed=1,return_ci8=False)
x=realism.apply(r['iq'],2e6,seed=1,filter_hz=1.75e6,dc_wander_lsb=0.5,iq_gain_db=0.3,iq_phase_deg=2)
sp=ss.analyse_stream(lambda: ss.iter_array(x),2e6)
f,P=sp['f'],sp['P']
db=10*np.log10(P[:len(P)//512*512].reshape(-1,512).mean(1)); ff=f[:len(P)//512*512].reshape(-1,512).mean(1)
print(np.round(db[::4],1)); print(np.round(ff[::4]/1e3))

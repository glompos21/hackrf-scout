import sys, time, numpy as np
sys.path.insert(0,'..'); sys.path.insert(0,'.')
import gen, sp_stream as ss
t=time.time(); r=gen.make('noise',2e6,1.0,0,seed=1,dc=6.0,return_ci8=False); print('gen',time.time()-t)
iq=r['iq']; tr=r['truth']
t=time.time(); sp=ss.analyse_stream(lambda: ss.iter_array(iq),2e6); print('analyse',time.time()-t)
P=sp['P']; f=sp['f']
print('dc',sp['dc'],'k',sp['k'],'keff',sp['k_eff'], 'N0 truth',tr['noise_psd_lsb2_per_hz'],'mean P',P.mean())
# ripple statistics
use=np.abs(f)<0.9e6
x=P[use]/P[use].mean()
print('std/mean fine bins', x.std(), 'expected 1/sqrt(keff)', 1/np.sqrt(sp['k_eff']))
for g in (2,4,8,16,64):
    m=len(x)//g*g
    y=x[:m].reshape(-1,g).mean(1)
    sf=g*g/(g+0.889*(g-1)+0.0556*(g-2)) if g>1 else 1
    print(g,'std',y.std(),'pred',1/np.sqrt(sp['k_eff']*sf))
print(sp['S'].shape)

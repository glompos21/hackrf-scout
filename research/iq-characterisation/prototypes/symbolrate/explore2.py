import sys
import numpy as np
sys.path.insert(0,'..')
import gen, frontend as fe
from explore1 import prep, linespec
kind=sys.argv[1]; rate=float(sys.argv[3]) if len(sys.argv)>3 else 2e6
for snr in [float(s) for s in sys.argv[2].split(',')]:
    r=gen.make(kind,rate,4.0,snr,seed=1,offset_hz=60e3,cfo_hz=3e3,dc=4.0,return_ci8=False)
    tr=r['truth']
    rec,fs,n_ch,info,b,fb=prep(r,rate)
    print('\n==',kind,snr,'fs',fs,'n',len(rec),'truth sym',tr['symbol_rate'],'bw',tr['occupied_bw_hz'],'det band',round(b['lo']),round(b['hi']),'neb',info['neb'])
    p=np.abs(rec)**2
    # discriminator
    d=np.angle(rec[1:]*np.conj(rec[:-1]))*fs/(2*np.pi)
    # gate: only samples whose smoothed power is above 1.5 n_ch
    for Ls in [1,4,8]:
        s=np.convolve(d,np.ones(Ls)/Ls,mode='valid')
        s=s-np.median(s)
        h,e=np.histogram(s,bins=40,range=(-60e3,60e3))
        print('Ls',Ls,'hist',' '.join('%d'%(v*100//h.sum()) for v in h))

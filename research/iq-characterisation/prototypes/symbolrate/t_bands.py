import sys, time
import numpy as np
sys.path.insert(0,'..')
import gen, frontend as fe
cases=[('ook',2e6,{}),('fsk2',2e6,{}),('qpsk',2e6,{}),('nfm',2e6,{}),('cw',2e6,{}),('am',2e6,{}),('wfm',2e6,{}),('gfsk',4e6,{}),('lora',2e6,{}),('noise',2e6,{})]
snr=float(sys.argv[1]) if len(sys.argv)>1 else 10
for kind,rate,kw in cases:
    r=gen.make(kind,rate,3.0,snr,seed=1,offset_hz=60e3,cfo_hz=3e3,dc=4.0,on_fraction=1.0,return_ci8=False,**kw)
    tr=r['truth']
    src=fe.ArraySource(r['iq'],rate)
    t=time.time(); sc=fe.scan_psd(src); t1=time.time()-t
    fb=fe.find_bands(sc,rate)
    b=fb['bands'][0] if fb['bands'] else None
    tb=[(round(a/1e3,1),round(b_/1e3,1)) for a,b_ in tr['bands_hz']] if tr['bands_hz'] else None
    bb=(round(b['lo']/1e3,1),round(b['hi']/1e3,1),round(b['centroid']/1e3,1)) if b else None
    print(kind, 'true center %.1fk'%(tr['center_hz']/1e3), 'tb', tb, '| nbands', len(fb['bands']), 'best', bb, 'sigma %.4f t=%.2fs'%(fb['sigma'],t1))

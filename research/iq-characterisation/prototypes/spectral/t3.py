import sys, time, numpy as np
sys.path.insert(0,'..'); sys.path.insert(0,'.')
import gen, sp_stream as ss, sp_feat as sf
cases=[('cw',2e6,1.0,{}),('am',2e6,1.0,{}),('nfm',2e6,1.0,{}),('ook',2e6,1.0,{}),('fsk2',2e6,1.0,{}),('lora',2e6,1.0,{})]
for kind,rate,dur,kw in cases:
  for snr in (0,5,10,20):
    for cond in (0,1):
        kws=dict(offset_hz=23e3)
        if cond: kws.update(dc=6.0,cfo_hz=3.5e3,on_fraction=0.2)
        try: r=gen.make(kind,rate,dur,snr,seed=1,return_ci8=False,**kws,**kw)
        except Exception as e: print(kind,e); continue
        tr=r['truth']
        sp=ss.analyse_stream(lambda: ss.iter_array(r['iq']),rate); F=sf.features(sp)
        g=lambda k,fmt='%.3g': (fmt%F[k]) if F.get(k) is not None else 'None'
        print('%-5s snr=%2d cond=%d B=%-7.0f obw=%-8s det=%s struct_z=%-7s lines=%-3s snr_est=%s lobes=%s flat=%s dcz=%s cen=%s (true %.0f)'%(kind,snr,cond,tr['occupied_bw_hz'],g('obw99_hz','%.0f'),F.get('detected'),g('struct_z','%.1f'),F['n_lines'],g('snr_est_db','%.1f'),g('n_lobes'),g('flat_db','%.2f'),g('dc_resid_z','%.1f'),g('obw_centre_hz','%.0f'),tr['center_hz']))

import sys, time, numpy as np
sys.path.insert(0,'..'); sys.path.insert(0,'.')
import gen, sp_stream as ss, sp_feat as sf
cases=[('noise',2e6,1.0,{}),('cw',2e6,1.0,{}),('am',2e6,1.0,{}),('nfm',2e6,1.0,{}),('wfm',2e6,1.0,{}),
 ('ook',2e6,1.0,{}),('fsk2',2e6,1.0,{}),('gfsk',4e6,1.0,{}),('qpsk',2e6,1.0,{}),('lora',2e6,1.0,{}),
 ('ofdm',10e6,0.5,{}),('hopper',10e6,0.5,{}),('pulsed',2e6,1.0,{}),('multi',2e6,1.0,{})]
snr=float(sys.argv[1]) if len(sys.argv)>1 else 15
for kind,rate,dur,kw in cases:
    r=gen.make(kind,rate,dur,snr,seed=0,return_ci8=False,**kw); tr=r['truth']
    t=time.time(); sp=ss.analyse_stream(lambda: ss.iter_array(r['iq']),rate); t1=time.time()-t
    t=time.time(); F=sf.features(sp); t2=time.time()-t
    g=lambda k,fmt='%.3g': (fmt%F[k]) if F.get(k) is not None else 'None'
    print('%-7s B=%-9s obw=%-9s n0ok=%s struct_z=%s lines=%s lobes=%s flat=%s ptm=%s edge=%s cpz=%s cplag=%s cpA=%s hopch=%s dwell=%s conc=%s t=%.2f+%.2f'%(
      kind, tr['occupied_bw_hz'] and '%.0f'%tr['occupied_bw_hz'], g('obw99_hz','%.0f'), F['floor_ok'], g('struct_z','%.1f'), F['n_lines'], g('n_lobes'), g('flat_db','%.2f'), g('ptm_db','%.1f'),
      g('edge_ratio','%.2f'), g('cp_z','%.0f'), g('cp_lag'), g('cp_A','%.3f'), g('hop_n_ch'), g('hop_dwell_s','%.2e'), g('hop_conc','%.2f'), t1,t2))

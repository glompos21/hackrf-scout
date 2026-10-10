import sys, time
sys.path.insert(0, '..')
import gen
for kind, rate, kw in [('ook',2e6,{}),('fsk2',2e6,{}),('qpsk',2e6,{}),('nfm',2e6,{}),('gfsk',4e6,{}),('cw',2e6,{})]:
    t=time.time(); r=gen.make(kind, rate, 5.0, 10.0, seed=1, return_ci8=False, **kw); dt=time.time()-t
    tr=r['truth']
    print(kind, rate, '%.2fs'%dt, 'sym', tr['symbol_rate'], 'bw', tr['occupied_bw_hz'], 'on', tr['on_fraction'], 'clip', tr['clip_fraction'], r['iq'].dtype, r['iq'].shape)

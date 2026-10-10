import sys, time
import numpy as np
sys.path.insert(0,'..')
import frontend as fe
rate=2e6; n=int(3e6)
rng=np.random.default_rng(0)
t=np.arange(n)/rate
x=(np.exp(2j*np.pi*(150e3+1234.5)*t)+ 0.5*np.exp(2j*np.pi*(700e3)*t) + (rng.standard_normal(n)+1j*rng.standard_normal(n))*0.1/np.sqrt(2)).astype(np.complex64)+ (3+2j)
src=fe.ArraySource(x,rate)
rec,fs,info=fe.channelize(src, 150e3, 50e3, 16, dc=3+2j)
print(fs, info, len(rec), n/16)
m=np.arange(len(rec))
# tone estimate
sp=np.fft.fft(rec*np.hanning(len(rec)))
k=np.argmax(abs(sp)); f=np.fft.fftfreq(len(rec),1/fs)[k]
print('tone at', f, 'expected', 150e3+1234.5-info['fc'], 'amp', np.abs(rec[1000:-1000]).mean(), 'std amp', np.abs(rec[1000:-1000]).std())
# noise power
x2=(rng.standard_normal(n)+1j*rng.standard_normal(n)).astype(np.complex64)/np.sqrt(2)  # power 1 per sample, N0 = 1/rate
src2=fe.ArraySource(x2,rate)
rec2,fs2,info2=fe.channelize(src2, 100e3, 50e3, 16)
print('noise pow', (np.abs(rec2[500:-500])**2).mean(), 'expected', info2['neb']/rate, 'neb', info2['neb'])

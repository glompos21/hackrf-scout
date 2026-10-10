import sys, time
import numpy as np
sys.path.insert(0,'..')
import gen, frontend as fe

def prep(r, rate):
    src=fe.ArraySource(r['iq'],rate)
    sc=fe.scan_psd(src); fb=fe.find_bands(sc,rate)
    b=fb['bands'][0]
    bw=1.3*(b['hi']-b['lo'])+2e3
    D=fe.choose_decimation(rate,bw)
    rec,fs,info=fe.channelize(src,b['centroid'],bw,D,dc=sc['dc'])
    n_ch=fb['n0']*info['neb']
    return rec,fs,n_ch,info,b,fb

def linespec(u, fs, nfft):
    u=u-u.mean()
    nseg=len(u)//nfft
    x=u[:nseg*nfft].reshape(nseg,nfft)*np.hanning(nfft)
    S=(np.abs(np.fft.rfft(x,axis=1))**2).mean(0)
    f=np.fft.rfftfreq(nfft,1/fs)
    return f,S,nseg

if __name__=='__main__':
    kind=sys.argv[1]; snr=float(sys.argv[2]); rate=float(sys.argv[3]) if len(sys.argv)>3 else 2e6
    kw={}
    r=gen.make(kind,rate,5.0,snr,seed=1,offset_hz=60e3,cfo_hz=3e3,dc=4.0,return_ci8=False,**kw)
    tr=r['truth']
    rec,fs,n_ch,info,b,fb=prep(r,rate)
    print(kind,snr,'fs',fs,'n',len(rec),'n_ch',n_ch,'truth sym',tr['symbol_rate'],'on',tr['on_fraction'])
    p=np.abs(rec)**2
    print('mean power/n_ch', p.mean()/n_ch)
    # smoothed envelope
    L=max(1,int(round(fs/ info['neb'])))
    a=np.convolve(p,np.ones(L)/L,mode='same')
    tau=max(1,int(round(fs/info['neb'])))
    u=(a[tau:]-a[:-tau])**2
    nfft=4096
    f,S,nseg=linespec(u,fs,nfft)
    med=np.median(S)
    k=np.argsort(S)[::-1][:8]
    print('edge-density spectrum top lines:',[(round(f[i],1),round(S[i]/med,1)) for i in k], 'nseg',nseg)
    # |y|^2 cyclic line
    f,S,nseg=linespec(p,fs,nfft)
    med=np.median(S)
    k=np.argsort(S)[::-1][:8]
    print('|y|^2 spectrum top lines:',[(round(f[i],1),round(S[i]/med,1)) for i in k])

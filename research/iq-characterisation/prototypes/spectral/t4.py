import sys, time
sys.path.insert(0,'.')
import harness as h
for case in h.CASES:
    for cond in ('clean','imp'):
        t=time.time()
        try:
            r=h.run_capture(case,10,0,cond)
        except Exception as e:
            print(case,cond,'ERR',repr(e)); continue
        F=r['feat']
        print('%-15s %-5s ok  gen=%.2f pass=%.2f feat=%.2f det=%s obw=%s lines=%s hop=%s/%s cpz=%s'%(case,cond,r['t_gen'],r['t_pass'],r['t_feat'],F.get('detected'),F.get('obw99_hz') and round(F['obw99_hz']),F['n_lines'],F.get('hop_n_ch'),F.get('hop_n_hops'),F.get('cp_z') and round(F['cp_z'])))

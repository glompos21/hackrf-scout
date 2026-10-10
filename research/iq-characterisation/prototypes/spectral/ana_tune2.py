import sys; sys.path.insert(0,'.')
import numpy as np, summ, harness
rows=summ.load(sys.argv[1])
feats=sys.argv[2].split(',')
conds=sys.argv[3].split(',') if len(sys.argv)>3 else ['clean','imp']
mode=sys.argv[4] if len(sys.argv)>4 else 'med'
SN=[0,5,10,15,20,30]
for ft in feats:
  for cond in conds:
    print('\n== %s  [%s]  (%s over seeds; n/a = None)'%(ft,cond,mode))
    print('%-15s'%'case'+''.join('%10d'%s for s in SN))
    for case in harness.CASES:
        if case.startswith('noise'):
            rr=[r for r in rows if r['case']==case and r['cond']==cond]
            a=summ.col(rr,ft); 
            print('%-15s'%case+'%10s'%('%.3g'%np.nanmedian(a) if np.isfinite(a).any() else 'n/a')+'   (noise: min %.3g max %.3g)'%(np.nanmin(a) if np.isfinite(a).any() else np.nan, np.nanmax(a) if np.isfinite(a).any() else np.nan))
            continue
        line='%-15s'%case
        for s in SN:
            rr=[r for r in rows if r['case']==case and r['cond']==cond and r['snr']==s]
            a=summ.col(rr,ft)
            if not len(a) or not np.isfinite(a).any(): line+='%10s'%'n/a'; continue
            if mode=='med': v=np.nanmedian(a); line+='%10.3g'%v
            elif mode=='min': line+='%10.3g'%np.nanmin(a)
            elif mode=='max': line+='%10.3g'%np.nanmax(a)
            elif mode=='nfin': line+='%10.2f'%np.mean(np.isfinite(a))
            elif mode=='mean': line+='%10.3g'%np.nanmean(a)
        print(line)

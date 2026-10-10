import sys, json; sys.path.insert(0,'.')
import numpy as np, summ, harness
rows=summ.load(sys.argv[1]); conds=sys.argv[2].split(',') if len(sys.argv)>2 else ['clean','imp']
refs=json.load(open('results/refs.json'))
SN=[0,5,10,15,20,30]
# reference: nominal (truth). For hopper the average-PSD span is compared with the band hull.
def nominal(case):
    r=refs[case]; 
    return r['hull'] if case.startswith('hopper') else r['nominal']
for cond in conds:
    print('\n== obw99 / nominal  [%s] median (frac of captures with a measurement)'%cond)
    print('%-15s %9s'%('case','nominal')+''.join('%14d'%s for s in SN))
    for case in harness.CASES:
        if case.startswith('noise'): continue
        line='%-15s %9.0f'%(case,nominal(case))
        for s in SN:
            rr=[r for r in rows if r['case']==case and r['cond']==cond and r['snr']==s]
            a=summ.col(rr,'obw99_hz')/nominal(case)
            fin=np.isfinite(a)
            line+='%9.2f(%3.1f)'%(np.nanmedian(a) if fin.any() else np.nan, fin.mean())
        print(line)

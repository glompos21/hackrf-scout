import sys; sys.path.insert(0,'.')
import numpy as np, summ
rows=summ.load('results/tune.jsonl')
print(len(rows))
# noise statistics
nz=[r for r in rows if r['case'].startswith('noise')]
for key in ('struct_z','dc_resid_z','cp_z','cp_A','hop_n_act','n_lines','floor_frac','floor_ok','detected','occ_frac','n0'):
    a=summ.col(nz,key); print('noise',key,'min %.3g med %.3g max %.3g'%(np.nanmin(a),np.nanmedian(a),np.nanmax(a)))
for sc in (1,4,16,64,256):
    a=np.array([r['feat']['struct_z_by_scale'].get(str(sc),np.nan) for r in nz]); print('scale',sc,'max z over noise: %.2f  (mean %.2f)'%(np.nanmax(a),np.nanmean(a)))

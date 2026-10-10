"""Single-feature separability (AUC, 0.5 = useless) between class groups, by SNR range.
usage: feature_auc.py file.jsonl"""
import sys, json, math
import numpy as np
import evaluate as E, flat

def auc(a, b):
    a = np.asarray([x for x in a if x == x]); b = np.asarray([x for x in b if x == x])
    if len(a) < 3 or len(b) < 3: return float('nan')
    allv = np.concatenate([a, b]); ranks = allv.argsort().argsort() + 1.0
    # average ties
    order = np.argsort(allv, kind='mergesort'); sv = allv[order]
    r = np.empty(len(allv)); i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    ra = r[:len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2.0) / (len(a) * len(b)))

GROUPS = {
    'const-env data (fsk2*,gfsk) vs analog FM (nfm_voice,wfm)': (['fsk2_rect','fsk2_gauss','fsk2_h1','fsk2_h2','gfsk'], ['nfm_voice','wfm']),
    'const-env data (fsk2*,gfsk) vs tone-FM + LoRa': (['fsk2_rect','fsk2_gauss','fsk2_h1','fsk2_h2','gfsk'], ['nfm_tone','lora','lora_sf9']),
    'PSK (bpsk,qpsk*) vs constant-envelope (cw,fm,fsk)': (['bpsk','qpsk','qpsk_50k'], ['cw','nfm_voice','wfm','fsk2_rect','fsk2_gauss','fsk2_h1','fsk2_h2','gfsk']),
    'OOK vs AM (voice+tone)': (['ook_pwm','ook_man','ook_nrz'], ['am_tone','am_voice']),
    'OOK vs PSK': (['ook_pwm','ook_man','ook_nrz'], ['bpsk','qpsk','qpsk_50k']),
    'carrier (cw) vs AM': (['cw'], ['am_tone','am_voice']),
    'cw vs fm': (['cw'], ['nfm_voice','wfm']),
}
FEATS = ['env_exc','env_V','top2_frac_f','peak3','disc_bc','disc_centre_frac','disc_ku','sigf_bw','abs_acfT','dmf_lr','dmf_nh','envf_lr','x2_delta','x4_delta','x2_lr','x4_lr','ook_sup','ook_n1','ook_sep','ook_duty','fskr_sup','fskr_n1']

if __name__ == '__main__':
    recs = [d for d in E.load(sys.argv[1]) if d['feat'].get('stage') == 'ok']
    rows = []
    for d in recs:
        o = flat.flatten(d['feat']); o['top2_frac_f'] = d['feat'].get('top2_frac', float('nan'))
        rows.append((d['variant'], d['snr'], o))
    for gname, (A, B) in GROUPS.items():
        print('\n##', gname)
        for lo, hi in ((10, 31), (0, 6)):
            print('   SNR %d..%d dB:' % (lo, hi - 1 if hi < 31 else 30), end='')
            res = []
            for ft in FEATS:
                a = [o.get(ft, float('nan')) for v, s, o in rows if v in A and lo <= s < hi]
                b = [o.get(ft, float('nan')) for v, s, o in rows if v in B and lo <= s < hi]
                x = auc(a, b)
                if x == x: res.append((abs(x - 0.5) * 2, ft, x))
            res.sort(reverse=True)
            print('  ' + '  '.join('%s %.2f' % (ft, x) for _, ft, x in res[:8]))

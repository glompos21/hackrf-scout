import sys
sys.path.insert(0, '.')
import numpy as np, summ, rules, eval_rules, harness
T = rules.load_thr()
ev = eval_rules.annotate(summ.load('results/eval.jsonl'), T)
lo = eval_rules.annotate(summ.load('results/lowsnr.jsonl'), T)
allr = ev + lo
SN = [-15, -12, -9, -6, -3, 0, 5, 10]
for cond in ('clean', 'imp'):
    print('\n=== correct-label rate by SNR incl. below 0 dB (cond=%s; -15..-3: 10 seeds, 0..10: 20 seeds). detect = not noise-only/unknown' % cond)
    print('%-14s %-8s' % ('case', 'truth') + ''.join('%11d' % s for s in SN))
    for case in harness.CASES:
        if harness.CASES[case][0] in ('noise', 'pulsed', 'lora') and case not in ('lora-sf7',): 
            if case not in ('lora-sf7',): continue
        line = '%-14s %-8s' % (case, '')
        tl = None
        for s in SN:
            rr = [r for r in allr if r['case'] == case and r['cond'] == cond and r['snr'] == s]
            if not rr:
                line += '%11s' % '-'; continue
            tl = rr[0]['true']
            ok = np.mean([r['pred'] == r['true'] for r in rr]); det = np.mean([r['pred'] not in ('noise-only', 'unknown') for r in rr])
            line += '  %3.0f%%/%3.0f%%' % (100 * ok, 100 * det)
        print(line.replace('%-8s' % '', '%-8s' % (tl or ''), 1))
print('\n(cell = correct-label% / detect%)')

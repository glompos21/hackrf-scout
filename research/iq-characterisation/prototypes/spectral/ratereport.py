import sys, json
sys.path.insert(0, '.')
import numpy as np, summ, harness, rules
T = rules.load_thr()
rows = [json.loads(l) for l in open('results/ratesweep.jsonl')]
rows = [r for r in rows if 'error' not in r]
refs = json.load(open('results/refs.json'))
print('=== OBW99 vs sample rate (same signal, same SNR; clean, seeds 100-104). median obw99 [Hz] and ratio to the 2 Msps (or lowest-rate) value at the same SNR; df = fine-bin width')
for case in ['nfm-voice', 'wfm', 'bpsk', 'lora-sf7', 'fsk2-rect', 'gfsk4']:
    for snr in (0, 10, 20):
        line = '%-10s snr %2d: ' % (case, snr)
        base = None
        for rate in (2e6, 4e6, 10e6, 20e6):
            rr = [r for r in rows if r['case'] == case and r['snr'] == snr and r['rate_run'] == rate]
            if not rr: continue
            a = summ.col(rr, 'obw99_hz')
            m = np.nanmedian(a)
            if base is None: base = m
            df = rr[0]['feat']['df']
            line += ' | %2.0f Msps: %9.0f (x%.2f, df %4.0f Hz, B/df %5.0f, nominal %.2f)' % (rate / 1e6, m, m / base, df, m / df, m / refs[case]['nominal'])
        print(line)
print('\n=== floor_ok / struct / classification vs rate (frozen rules)')
for case in ['nfm-voice', 'bpsk', 'lora-sf7', 'gfsk4']:
    for rate in (2e6, 4e6, 10e6, 20e6):
        rr = [r for r in rows if r['case'] == case and r['rate_run'] == rate and r['snr'] == 10]
        if not rr: continue
        labs = [rules.classify(r['feat'], T)[0] for r in rr]
        from collections import Counter
        print('%-10s %2.0f Msps: %s  floor_ok %d/%d  t_pass %.2fs (n=%d samples)' % (case, rate / 1e6, dict(Counter(labs)), sum(r['feat']['floor_ok'] for r in rr), len(rr), np.mean([r['t_pass'] for r in rr]), rr[0]['feat']['n']))

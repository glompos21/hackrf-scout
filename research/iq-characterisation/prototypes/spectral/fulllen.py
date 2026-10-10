"""Full-length confirmation: 5 s captures streamed from a ci8 file (the real operating point), frozen rules.
Generated in a child process; analysed from disk in chunks.  Seeds 100-101 (eval seeds)."""
import os, sys, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, sp_stream, sp_feat, rules
T = rules.load_thr()
TMP = os.path.join(HERE, 'tmp'); os.makedirs(TMP, exist_ok=True)
SPECS = [  # kind, rate, kw, snr, offset
    ('noise', 10e6, {}, 0, 0.0), ('noise', 2e6, {}, 0, 0.0),
    ('qpsk', 10e6, {}, 10, 40e3), ('ofdm', 10e6, {}, 10, 40e3), ('hopper', 10e6, {}, 10, 40e3), ('gfsk', 10e6, {}, 10, 40e3),
    ('nfm', 2e6, {}, 10, 20e3), ('cw', 2e6, {}, 10, 17e3), ('lora', 2e6, {'sf': 7}, 10, 20e3), ('ook', 2e6, {}, 10, 15e3),
    ('ofdm', 10e6, {}, 0, 40e3), ('hopper', 10e6, {}, 0, 40e3),
]
out = []
for kind, rate, kw, snr, off in SPECS:
    for seed in (100, 101):
        path = os.path.join(TMP, 'x.ci8')
        code = ("import sys; sys.path.insert(0, %r); import gen; kw=dict(offset_hz=%r, dc=6.0, cfo_hz=2500.0); kw.update(%r); r = gen.make(%r, %r, 5.0, %r, seed=%r, **kw); gen.write_ci8(r, %r); import json; t=r['truth']; print(json.dumps(dict(bw=t['occupied_bw_hz'], clip=t['clip_fraction'])))" %
                (os.path.dirname(HERE), off, kw, kind, rate, float(snr), seed, path))
        if kind == 'noise':
            code = code.replace("kw=dict(offset_hz=%r, dc=6.0, cfo_hz=2500.0)" % off, "kw=dict(dc=6.0)")
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        if r.returncode:
            print('gen failed', kind, r.stderr[-300:]); continue
        meta = json.loads(r.stdout.strip().splitlines()[-1])
        t0 = time.process_time(); w0 = time.time()
        sp = sp_stream.analyse_stream(lambda: sp_stream.iter_ci8_file(path), rate)
        F = sp_feat.features(sp)
        cpu = time.process_time() - t0; wall = time.time() - w0
        lab, conf, why = rules.classify(F, T)
        out.append(dict(kind=kind, rate=rate, snr=snr, seed=seed, label=lab, conf=conf, obw=F.get('obw99_hz'), nominal=meta['bw'],
                        n_lines=F['n_lines'], cpu=cpu, wall=wall, struct_z=F.get('struct_z'), clip=meta['clip']))
        print('%-7s %4.0f Msps snr %2d seed %d -> %-10s conf %.2f obw %-9s (nominal %-9s) lines %-3d struct_z %-7.1f cpu %.1fs wall %.1fs' %
              (kind, rate / 1e6, snr, seed, lab, conf, ('%.0f' % F['obw99_hz']) if F.get('obw99_hz') else '-', ('%.0f' % meta['bw']) if meta['bw'] else '-',
               F['n_lines'], F.get('struct_z') or 0, cpu, wall), flush=True)
        os.remove(path)
json.dump(out, open(os.path.join(HERE, 'results', 'fulllen.json'), 'w'), indent=1)

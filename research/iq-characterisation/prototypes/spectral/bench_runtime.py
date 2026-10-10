"""Runtime / memory of the whole spectral family on a 5 s @ 10 Msps ci8 file (100 MB), streamed in 1M-sample chunks.
Generation happens in a separate process so the RSS number is the analysis only."""
import os, sys, time, json, subprocess, resource
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
TMP = os.path.join(HERE, 'tmp'); os.makedirs(TMP, exist_ok=True)

def gen_file(kind, kw, path, rate=10e6, secs=5.0, snr=12.0, **p):
    code = ("import sys; sys.path.insert(0, %r); import gen; r = gen.make(%r, %r, %r, %r, seed=7, **%r); gen.write_ci8(r, %r)" %
            (os.path.dirname(HERE), kind, rate, secs, snr, dict(kw, **p), path))
    subprocess.run([sys.executable, '-c', code], check=True)

def analyse(path, rate, stride=1, want_tf=True, overlap=0.5):
    import numpy as np, sp_stream, sp_feat
    t0 = time.perf_counter(); c0 = time.process_time()
    sp = sp_stream.analyse_stream(lambda: sp_stream.iter_ci8_file(path), rate, stride=stride, want_tf=want_tf, overlap=overlap)
    t1 = time.perf_counter(); c1 = time.process_time()
    F = sp_feat.features(sp, hop=want_tf)
    t2 = time.perf_counter(); c2 = time.process_time()
    return dict(wall_stream=t1 - t0, cpu_stream=c1 - c0, wall_feat=t2 - t1, cpu_feat=c2 - c1,
                maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, k=sp['k'],
                label=None, obw=F.get('obw99_hz'), n_lines=F['n_lines'], hop=F.get('hop_n_ch'), cp_z=F.get('cp_z'))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'child':
        path, rate, stride, tf, ov = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
        print(json.dumps(analyse(path, rate, stride, bool(tf), ov), default=float))
        sys.exit(0)
    cases = [('qpsk', {}, 'qpsk'), ('ofdm', {}, 'ofdm'), ('hopper', {}, 'hopper'), ('noise', {}, 'noise')]
    out = {}
    for kind, kw, name in cases:
        path = os.path.join(TMP, name + '.ci8')
        if not os.path.exists(path):
            gen_file(kind, kw, path)
        for stride, tf, ov in ((1, 1, 0.5), (1, 0, 0.5), (1, 1, 0.0), (4, 1, 0.5), (4, 1, 0.0)):
            r = subprocess.run([sys.executable, __file__, 'child', path, '10e6', str(stride), str(tf), str(ov)], capture_output=True, text=True)
            d = json.loads(r.stdout.strip().splitlines()[-1])
            out['%s|stride%d|tf%d|ov%.1f' % (name, stride, tf, ov)] = d
            print('%-7s stride=%d tf=%d ov=%.1f  stream wall %.2fs cpu %.2fs | features wall %.2fs cpu %.2fs | peak RSS %.0f MB | segments %d' %
                  (name, stride, tf, ov, d['wall_stream'], d['cpu_stream'], d['wall_feat'], d['cpu_feat'], d['maxrss_mb'], d['k']), flush=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'runtime.json'), 'w'), indent=1)

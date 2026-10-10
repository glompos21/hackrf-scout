"""Analyse one ci8 file in a fresh process: python3 analyze_file.py file.ci8 rate [prior_fc prior_bw]  -> timing + peak RSS."""
import sys, os, time, resource
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ampfeat as A
path, rate = sys.argv[1], float(sys.argv[2])
prior = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else None
def hwm():
    for line in open('/proc/self/status'):
        if line.startswith('VmHWM'):
            return int(line.split()[1]) / 1e3
    return float('nan')
rss0 = hwm()
t = time.perf_counter(); R = A.analyze(path, rate, prior=prior); dt = time.perf_counter() - t
rss = hwm()
tm = R['timing']
print('%-12s %4.0f MB file | total %5.1fs  (pass1 %.1f  band %.2f  pass2 %.1f  hmm+feats %.1f) | peak RSS %4.0f MB (%.0f at start) | channel %s bw %.0f Hz, %d ticks of %.1f us | duty %.3f runs %d' % (
    os.path.basename(path), os.path.getsize(path) / 1e6, dt, tm['pass1'], tm['band'], tm['pass2'], tm['feats'], rss, rss0,
    R['channel']['src'], R['channel']['bw'], R['channel']['T'], R['channel']['tick_s'] * 1e6, R['duty'], R['runs']['n_runs']))

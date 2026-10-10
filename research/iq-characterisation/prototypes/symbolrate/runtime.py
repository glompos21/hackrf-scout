"""5 s @ 10 Msps (and 2/4 Msps) runtime + memory, single process pinned to one core when taskset exists."""
import sys, os, json, subprocess, shutil, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scen, gen
tmp = '/tmp/claude-0/-home-user-hackrf-scout/f99c7e11-f52c-5034-af0a-858cb1074c59/scratchpad/iqchar/symbolrate/tmp_rt'
os.makedirs(tmp, exist_ok=True)
cases = [
    ('fsk2_rect', 10e6, 5.0, 'cont', 15), ('ook_pwm', 10e6, 5.0, 'cont', 15), ('qpsk_1M', 10e6, 5.0, 'cont', 15), ('gfsk', 10e6, 5.0, 'cont', 15),
    ('noise', 10e6, 5.0, 'cont', 0), ('ook_pwm', 10e6, 5.0, 'gated', 15), ('nfm_voice', 2e6, 5.0, 'cont', 15), ('fsk2_rect', 2e6, 5.0, 'cont', 15),
    ('ook_pwm', 2e6, 5.0, 'gated', 15), ('qpsk', 2e6, 5.0, 'cont', 15),
]
res_all = []
for name, rate, secs, cond, snr in cases:
    kind = name
    kw = {}
    if name == 'qpsk_1M': kind = 'qpsk'; kw = dict(symrate=1e6)
    r = gen.make(kind if kind in gen.KINDS else scen.V[name][0], rate, secs, float(snr), seed=7, offset_hz=40e3, cfo_hz=2e3, dc=4 + 3j,
                 on_fraction=(1.0 if cond == 'cont' else 0.2), return_ci8=True, **(kw or ({} if name in gen.KINDS else scen.V[name][3])))
    path = os.path.join(tmp, '%s_%gM_%s.ci8' % (name, rate / 1e6, cond))
    open(path, 'wb').write(r['ci8'])
    del r
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime_child.py'), path, str(rate)]
    if shutil.which('taskset'): cmd = ['taskset', '-c', '0'] + cmd
    t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True); 
    line = [l for l in p.stdout.splitlines() if l.startswith('{')]
    if not line:
        print('FAILED', name, p.stderr[-500:]); continue
    d = json.loads(line[-1]); d.update(name=name, rate=rate, secs=secs, cond=cond, file_mb=os.path.getsize(path) / 1e6)
    res_all.append(d)
    print('%-10s %5.1fM %-5s file %5.0f MB | wall %5.2fs cpu %5.2fs rss %4.0f MB | %-8s baud %s | %s' % (name, rate / 1e6, cond, d['file_mb'], d['wall'], d['cpu'], d['rss_mb'], d['label'], d['baud'], d['timings']), flush=True)
    os.remove(path)
json.dump(res_all, open('results/runtime.json', 'w'), indent=1)

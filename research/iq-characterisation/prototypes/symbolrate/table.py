import json, sys, collections
import numpy as np
path = sys.argv[1]
cols = sys.argv[2].split(',')
cond = sys.argv[3] if len(sys.argv) > 3 else 'cont'
variants = sys.argv[4].split(',') if len(sys.argv) > 4 else None
def get(d, key):
    f = d['feat']
    if key == 'top_env_f': return f['env_lines'][0]['f'] if f.get('env_lines') else None
    if key == 'top_env_ratio': return f['env_lines'][0]['ratio'] if f.get('env_lines') else None
    if key == 'top_env_rel': return f['env_lines'][0]['f']/f['band_bw'] if f.get('env_lines') else None
    if key == 'dm_ratio': return max([x['ratio'] for x in f.get('dm_lines', [])] or [None]) if f.get('dm_lines') else None
    if key == 'dm_f':
        l = f.get('dm_lines', [])
        return max(l, key=lambda x: x['ratio'])['f'] if l else None
    if key == 'dm_rel':
        l = f.get('dm_lines', [])
        return max(l, key=lambda x: x['ratio'])['f']/f['band_bw'] if l else None
    if key == 'x2d': return f['x2_line']['delta'] if 'x2_line' in f else None
    if key == 'x4d': return f['x4_line']['delta'] if 'x4_line' in f else None
    if key == 'x2r': return f['x2_line']['ratio'] if 'x2_line' in f else None
    if key == 'x4r': return f['x4_line']['ratio'] if 'x4_line' in f else None
    if key == 'ook_T': return 1/f['ook']['T'] if f.get('ook') and 'T' in f['ook'] else None
    if key == 'ook_sup': return f['ook']['support'] if f.get('ook') and 'support' in f['ook'] else None
    if key == 'ook_sep': return f['ook']['sep'] if f.get('ook') else None
    if key == 'ook_lo': return f['ook']['lo_level'] if f.get('ook') else None
    if key == 'ook_n1': return f['ook']['frac_n1'] if f.get('ook') and 'frac_n1' in f['ook'] else None
    return f.get(key)
rows = collections.defaultdict(list)
for l in open(path):
    d = json.loads(l)
    if 'error' in d or d['cond'] != cond: continue
    if variants and d['variant'] not in variants: continue
    rows[(d['variant'], d['snr'])].append(d)
print('%-14s %4s %2s | ' % ('variant', 'snr', 'n') + ' | '.join('%11s' % c[:11] for c in cols))
for (v, s), ds in sorted(rows.items(), key=lambda kv: (list(sys.modules['__main__'].__dict__.get('ORDER', [])), kv[0][0], kv[0][1])):
    out = []
    for c in cols:
        vals = [get(d, c) for d in ds]
        vals = [x for x in vals if x is not None and not isinstance(x, (list, dict, str))]
        if not vals: out.append('%11s' % '-'); continue
        a = np.array(vals, float)
        if len(a) == 1: out.append('%11.3g' % a[0])
        else: out.append('%5.3g..%-5.3g' % (a.min(), a.max()))
    print('%-14s %4.0f %2d | ' % (v, s, len(ds)) + ' | '.join(out))

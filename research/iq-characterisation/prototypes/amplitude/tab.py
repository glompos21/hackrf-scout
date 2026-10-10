"""Tiny numpy-only table helpers for the jsonl rows."""
import json
import math
import numpy as np

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def col(rows, key, default=float('nan')):
    return np.array([float(r.get(key, default)) if r.get(key, default) is not None else float('nan') for r in rows], float)

def sel(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            rv = r.get(k)
            if isinstance(v, (list, tuple, set)):
                if rv not in v: ok = False; break
            elif callable(v):
                if not v(rv): ok = False; break
            elif rv != v:
                ok = False; break
        if ok: out.append(r)
    return out

def auc(pos, neg):
    """P(pos > neg) + 0.5 P(=) (Mann-Whitney), nan-safe; returns nan when a side is empty."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind='mergesort'); ranks = np.empty(len(allv)); 
    # average ranks for ties
    sv = allv[order]; r = np.arange(1, len(allv) + 1, dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1; i = j + 1
    ranks[order] = r
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))

def med(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else float('nan')

def q(v, p):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return float(np.percentile(v, p)) if len(v) else float('nan')

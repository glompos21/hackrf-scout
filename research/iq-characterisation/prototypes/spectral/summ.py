"""Small helpers to load and slice the jsonl feature files (numpy only)."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if 'error' in d:
                continue
            rows.append(d)
    return rows


def val(r, key, default=None):
    """feature lookup: 'feat.xxx' / 'truth.xxx' / top-level; None if missing."""
    if key.startswith('truth.'):
        return r['truth'].get(key[6:], default)
    if key in ('case', 'snr', 'seed', 'cond'):
        return r[key]
    return r['feat'].get(key, default)


def group(rows, keyfn):
    g = defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    return g


def col(rows, key):
    out = []
    for r in rows:
        v = val(r, key)
        out.append(np.nan if v is None else float(v))
    return np.array(out, float)


def frac(rows, pred):
    if not rows:
        return float('nan')
    return float(np.mean([bool(pred(r)) for r in rows]))


def q(a, p):
    a = a[~np.isnan(a)]
    return float(np.percentile(a, p)) if len(a) else float('nan')

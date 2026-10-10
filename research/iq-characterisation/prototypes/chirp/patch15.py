a = open('analyze.py').read()
i = a.index('def label_of(')
new = '''def verdict(res):
    """Turn the raw stage outputs into labels with a confidence tier and 'unknown' when nothing passes.
    chirp tiers   high   : lag-line + occupied band consistent with a LoRa bandwidth + dechirp tone in many windows
                          (preamble run >= 6 or >= 20 hit windows)
                  medium : lag-line + band consistent + >= 3 dechirp windows
                  low    : lag-line only (band not measurable, or no dechirp support)
    pulsed tiers  high   : >= 50 in-train spacings and >= 90 % of the pulses on the PRI grid were found
                  medium : >= 15 spacings          low : otherwise"""
    c, p = res.get('chirp', {}), res.get('pulsed', {})
    out = dict(chirp=None, pulsed=None, chirp_pulse=None)
    if c.get('flag'):
        b = c['best']
        dc = c.get('dechirp') or {}
        conf = 'low'
        if b.get('band_confirmed') and dc:
            if dc.get('preamble_run', 0) >= 6 or dc.get('n_hits', 0) >= 20:
                conf = 'high'
            elif dc.get('n_hits', 0) >= 3:
                conf = 'medium'
        out['chirp'] = dict(confidence=conf, bw_hz=b['bw_final'], sf=b['sf_final'], centre_hz=b['centre'],
                            sym_time_s=c.get('sym_time_s'), direction=b['dir'], band_confirmed=bool(b.get('band_confirmed')))
    if p.get('flag'):
        info = p['info']
        comp = p.get('completeness', 0.0)
        ncl = info.get('n_cluster', 0)
        conf = 'high' if (ncl >= 50 and comp >= 0.9) else ('medium' if ncl >= 15 else 'low')
        out['pulsed'] = dict(confidence=conf, pri_s=info['pri'], width_s=p.get('width_s') if comp >= 0.9 else None,
                             width_valid=bool(comp >= 0.9), duty=info['duty'], n_spacings=ncl, jitter=info['jitter'],
                             completeness=comp)
        cp = p.get('chirp_pulse')
        if cp and abs(cp['t']) > 6 and cp['frac_same_sign'] > 0.8:
            out['chirp_pulse'] = dict(slope_hz_per_s=cp['slope_hz_per_s'], swept_hz=cp['swept_hz'], t=cp['t'])
    return out


def label_of(res):
    v = verdict(res)
    res['verdict'] = v
    parts = []
    if v['chirp']:
        parts.append('chirp-like(%s)' % v['chirp']['confidence'])
    if v['pulsed']:
        parts.append('pulsed(%s)' % v['pulsed']['confidence'])
        if v['chirp_pulse']:
            parts.append('chirp-in-pulse')
    return '+'.join(parts) if parts else 'unknown'
'''
a = a[:i] + new
open('analyze.py', 'w').write(a)
h = open('harness.py').read()
h = h.replace("chirp=jclean(res['chirp']), pulsed=jclean(res['pulsed']), label=res['label'],", "chirp=jclean(res['chirp']), pulsed=jclean(res['pulsed']), label=res['label'], verdict=jclean(res['verdict']),")
open('harness.py', 'w').write(h)

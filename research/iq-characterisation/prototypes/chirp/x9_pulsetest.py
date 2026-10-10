import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-hackrf-scout/f99c7e11-f52c-5034-af0a-858cb1074c59/scratchpad/iqchar')
import numpy as np, gen
from chirp import analyze as an, frontend as fe, cases
cfg = an.load_cfg()
for name in ('pulsed_10us_1ms', 'pulsed_lfm_10us', 'pulsed_5us_2ms', 'pulsed_20us_jit', 'ook_pwm', 'ook_manchester', 'hopper', 'gfsk_250k', 'lora_sf7_125'):
    for snr in (20, 10):
        spec = cases.SPECS[name]
        gk, imp, rng = cases.case_params(name, snr, 100)
        r = gen.make(spec['kind'], spec['rate'], 5.0, float(snr), seed=100, return_ci8=False, **gk)
        meta = cases.sweep_meta(r['truth'], rng)
        res = an.characterise(fe.ArraySource(r['iq'], spec['rate']), meta=meta, cfg=cfg, want_css=False)
        p = res['pulsed']
        tl = cases.truth_labels(name, r['truth'])
        info = p.get('info', {})
        print('%-18s snr %2d | pulsed=%-5s n_ev=%5d pri=%s w=%s | cp=%s | truth pw3=%s pri=%s' % (name, snr, p['flag'], p['n_events'],
              ('%.1fus' % (info['pri'] * 1e6)) if info.get('pri') else None, ('%.2fus' % (p['width_s'] * 1e6)) if p.get('width_s') else None,
              ('mu=%.3g t=%.1f' % (p['chirp_pulse']['slope_hz_per_s'], p['chirp_pulse']['t'])) if p.get('chirp_pulse') else None,
              ('%.2fus' % (tl.get('pw3db_s', 0) * 1e6)) if 'pw3db_s' in tl else '-', ('%.0fus' % (tl['pri_s'] * 1e6)) if 'pri_s' in tl else '-'))
        if not p['flag'] and p['n_events'] > 3:
            print('     why:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}, p.get('pri'))

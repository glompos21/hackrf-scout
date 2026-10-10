"""Truth labels for the rule evaluation (derived from the generator truth only)."""
import harness

NARROW_MAX = 25e3      # nominal B below this: narrow
WIDE_MIN = 500e3       # nominal B at/above this: wide
GUARD = (0.7, 1.45)    # nominal B within these multiples of a boundary -> "don't care" (a boundary is a convention)


def truth_label(case, nominal_bw):
    kind = harness.CASES[case][0]
    if kind == 'noise':
        return 'noise-only', True
    if kind == 'cw':
        return 'carrier', True
    if kind == 'hopper':
        return 'hopper', True
    if kind == 'ofdm':
        return 'ofdm', True
    b = nominal_bw
    for edge in (NARROW_MAX, WIDE_MIN):
        if GUARD[0] * edge <= b <= GUARD[1] * edge:
            care = False
            break
    else:
        care = True
    lab = 'narrow' if b < NARROW_MAX else ('wide' if b >= WIDE_MIN else 'medium')
    return lab, care

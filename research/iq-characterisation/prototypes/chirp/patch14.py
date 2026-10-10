g = open('gated.py').read()
a = g.index("    med = float(np.median(exs[big]))")
b = g.index("    wide = np.convolve(exs, np.ones(9) / 9.0, mode='same')")
new = '''    wide = np.convolve(exs, np.ones(9) / 9.0, mode='same')
    # plateau level: start from the strongest smoothed bin and iterate (region above 25 % of the plateau -> median of the
    # region = new plateau); the iteration ignores skirts / sidelobes at high SNR and spectral horns on a flat top
    plateau = float(wide.max())
    for _ in range(3):
        thr = max(4.0 * sd, 0.25 * plateau)
        mask = exs > thr
        dil = np.convolve(mask.astype(np.int8), np.ones(7, np.int8), mode='same') >= 1             # closing: dilate ...
        mask = mask | (np.convolve(dil.astype(np.int8), np.ones(7, np.int8), mode='same') >= 7)    # ... then erode
        if mask.sum() < 3:
            return out
        plateau = float(np.median(exs[mask]))
    med = plateau
'''
g = g[:a] + new + g[b:]
g = g.replace("    wide = np.convolve(exs, np.ones(9) / 9.0, mode='same')\n    k = int(np.argmax(np.where(mask, wide, -1e30)))", "    k = int(np.argmax(np.where(mask, wide, -1e30)))")
open('gated.py', 'w').write(g)

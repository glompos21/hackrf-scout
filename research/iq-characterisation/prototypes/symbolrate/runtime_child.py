"""Child process: analyse one ci8 file with the streaming sources; prints one JSON line (time, CPU time, peak RSS)."""
import sys, os, json, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frontend as fe, features, classify as C
path, rate = sys.argv[1], float(sys.argv[2])
t0 = time.time(); c0 = time.process_time()
src = fe.Ci8FileSource(path, rate)
tim = {}
f = features.analyze(src, rate, timings=tim)
out = C.classify(f)
wall = time.time() - t0; cpu = time.process_time() - c0
print(json.dumps(dict(path=os.path.basename(path), wall=wall, cpu=cpu, rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                      label=out['label'], baud=out['baud_hz'], bound=out['bound_rel'], timings={k: round(v, 2) for k, v in tim.items()},
                      n=f.get('n'), n_rec=f.get('n_rec'), n_feat=f.get('n_feat'), early=f.get('early_stop'), blk=[f.get('blocks_active'), f.get('blocks_seen')])))

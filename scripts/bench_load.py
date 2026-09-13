"""Warm-cache model load time by load mode (model + context creation), via the cffi driver."""
import sys, os, time, json, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from driver import Engine
LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
out = {}
for mode in ['auto', 'none', 'mlock', 'mmap+mlock']:
    times = []
    for _ in range(3):
        t = time.perf_counter()
        e = Engine(os.path.join(LAB, 'models', 'Qwen3-0.6B-Q8_0.gguf'), os.path.join(LAB, 'bin', 'cpu'), n_threads=8, n_ctx=2048, load_mode=mode)
        toks = e.tokenize('Hello', add_special=False)
        e.greedy_loop(toks, 1)
        times.append(time.perf_counter() - t)
        e.close()
    out[mode] = dict(mean=statistics.mean(times), sd=statistics.pstdev(times), samples=times)
    print(f'{mode:12s} load+ctx+first token: {statistics.mean(times):.2f} s  ({[round(x,2) for x in times]})')
json.dump(out, open(os.path.join(LAB, 'results', 'load_modes.json'), 'w'), indent=1)

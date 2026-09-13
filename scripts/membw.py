"""Achievable DRAM read bandwidth. N processes each hold a private 512 MB float64 buffer; a Barrier
starts every pass at the same instant and the aggregate is (N * bytes) / wall time of the slowest process.
Not STREAM, but a fair upper bound for what a decode step can pull through the memory controller."""
import numpy as np, time, multiprocessing as mp

def worker(n_mb, passes, barrier, q):
    a = np.ones(n_mb * 1024 * 1024 // 8, dtype=np.float64); a.sum()
    times = []
    for _ in range(passes):
        barrier.wait()
        t = time.perf_counter(); a.sum(); times.append(time.perf_counter() - t)
        barrier.wait()
    q.put(times)

if __name__ == '__main__':
    n_mb, passes = 512, 6
    for nproc in [1, 2, 4, 6, 8, 12]:
        q = mp.Queue(); b = mp.Barrier(nproc)
        ps = [mp.Process(target=worker, args=(n_mb, passes, b, q)) for _ in range(nproc)]
        [p.start() for p in ps]
        allt = [q.get() for _ in ps]; [p.join() for p in ps]
        per_pass = [max(t[i] for t in allt) for i in range(passes)]   # slowest process bounds the pass
        agg = nproc * n_mb * 1024 * 1024 / min(per_pass)
        print(f'{nproc:2d} procs x {n_mb} MB: aggregate {agg/1e9:6.1f} GB/s (best pass), median {nproc*n_mb*1048576/np.median(per_pass)/1e9:6.1f} GB/s')

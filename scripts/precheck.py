"""Pre-checks to run on any machine before benchmarking or tuning an inference setup.
Prints what the hardware can do in the units that matter (GB/s, cores, ISA, VRAM) and flags the traps that
cost me days: battery power caps, hybrid cores, a single DIMM, a busy machine, a GPU that is power capped.
No arguments needed. Optional: --model path.gguf to print the fit arithmetic for one file.
Windows first; the Linux/macOS branches are best effort and say so."""
import os, sys, platform, subprocess, json, time, re, shutil, multiprocessing as mp

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BIN = os.path.join(LAB, 'bin', 'cpu')
WIN = platform.system() == 'Windows'


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ''


def ps(cmd):
    return sh(['powershell', '-NoProfile', '-Command', cmd])


def section(t):
    print(f"\n== {t}")


def bw_worker(n_mb, passes, barrier, q):
    import numpy as np
    a = np.ones(n_mb * 1024 * 1024 // 8, dtype=np.float64); a.sum()
    times = []
    for _ in range(passes):
        barrier.wait()
        t = time.perf_counter(); a.sum(); times.append(time.perf_counter() - t)
        barrier.wait()
    q.put(times)


def bandwidth(nproc, n_mb=256, passes=5):
    """Same method as membw.py: private buffers, a barrier per pass, aggregate over the slowest process."""
    q = mp.Queue(); b = mp.Barrier(nproc)
    procs = [mp.Process(target=bw_worker, args=(n_mb, passes, b, q)) for _ in range(nproc)]
    [p.start() for p in procs]
    allt = [q.get() for _ in procs]; [p.join() for p in procs]
    per_pass = [max(t[i] for t in allt) for i in range(passes)]
    return nproc * n_mb * 1048576 / min(per_pass) / 1e9


def main():
    warn = []
    n_log = os.cpu_count() or 1

    section('OS and power')
    print(f"{platform.system()} {platform.release()} ({platform.machine()})")
    if WIN:
        bat = ps("Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1 PowerOnline | ConvertTo-Json")
        if bat.strip():
            on_ac = bool(json.loads(bat).get('PowerOnline'))
            print(f"battery present, on AC: {on_ac}")
            if not on_ac:
                warn.append('ON BATTERY: the GPU power limit drops to a fraction of its default (24 W of 55 on my laptop; GPU decode fell 3 to 5x, CPU decode and prompt speed did not change). Plug in before measuring anything on the GPU.')
        print(ps('powercfg /getactivescheme').strip())
    else:
        print('power check: not implemented for this OS; on a laptop, confirm AC and the performance profile by hand')

    section('CPU')
    n_phys = None
    if WIN:
        c = json.loads(ps("Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json") or '{}')
        name = (c.get('Name') or '?').strip(); n_phys = c.get('NumberOfCores')
        print(f"{name}  physical cores={n_phys} logical={c.get('NumberOfLogicalProcessors')}")
        if re.search(r'i[3579]-1[2-9]\d{2,3}|Core\s*Ultra|Core\s*[3579]\s*\d{3}', name):
            warn.append(f'HYBRID CPU: the {n_phys} "physical cores" include E-cores. llama.cpp defaults -t to all of them; set -t to the P-core count and measure.')
    else:
        print(f"logical cpus={n_log}; lscpu (Linux) or sysctl hw (macOS) for the P/E split")
    try:
        import cpuinfo
        flags = set(cpuinfo.get_cpu_info().get('flags', []))
        isa = [f for f in ['sse4_2', 'avx', 'avx2', 'fma', 'f16c', 'avx_vnni', 'avx512f', 'avx512_vnni', 'avx512_bf16', 'amx_int8', 'neon', 'asimd', 'sve'] if f in flags]
        print('ISA:', ' '.join(isa) or '(none of the interesting flags)', '(py-cpuinfo does not list AVX-VNNI; the kernel DLL below tells you)')
        if not ({'avx2', 'neon', 'asimd'} & flags):
            warn.append('NO AVX2: ggml will load the sandybridge (AVX) or sse42/x64 kernels. Prompt processing 2 to 3x slower; decode still limited by bandwidth.')
        if not ({'avx', 'neon', 'asimd'} & flags):
            warn.append('NO AVX: the K-quant and IQ formats lose their fast dot products on these kernels. Prefer Q4_0 or Q8_0 files here; measure before trusting a Q4_K_M.')
    except ImportError:
        print('ISA: pip install py-cpuinfo for the flag list, or read the system_info line llama-bench prints')
    if os.path.isdir(BIN):
        variants = sorted(f[9:-4] for f in os.listdir(BIN) if f.startswith('ggml-cpu-') and f.endswith('.dll'))
        print(f"ggml-cpu variants in bin/cpu: {', '.join(variants)}; the loader scores each and keeps the best")

    section('Memory')
    peak = None
    if WIN:
        m = json.loads(ps("Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed, ConfiguredClockSpeed, DataWidth | ConvertTo-Json") or '[]')
        if isinstance(m, dict): m = [m]
        if m:
            total = sum(int(x['Capacity']) for x in m) / 2**30; chans = len(m)
            speed = max((x.get('ConfiguredClockSpeed') or x.get('Speed') or 0) for x in m); width = m[0].get('DataWidth') or 64
            peak = speed * 1e6 * (width / 8) * chans / 1e9
            print(f"{chans} DIMM(s), {total:.0f} GB total, {speed} MT/s, {width}-bit each: theoretical {peak:.1f} GB/s")
            if chans == 1:
                warn.append('SINGLE CHANNEL: one DIMM populated. A second DIMM is the cheapest 2x for decode there is.')
        free = ps("(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory").strip()
        if free: print(f"free now: {int(free) / 2**20:.1f} GB")
    else:
        print('DIMM count and speed: dmidecode -t memory (Linux) or system_profiler SPMemoryDataType (macOS)')
    try:
        import numpy  # noqa
        best = 0
        for n in sorted({1, 4, min(8, n_log)}):
            bw = bandwidth(n); best = max(best, bw)
            print(f"streaming read, {n} process(es): {bw:5.1f} GB/s")
        if peak: print(f"best is {100 * best / peak:.0f}% of theoretical; 60 to 80% is normal for this kind of test")
        print(f"decode ceilings at that bandwidth: 633 MB model {best / 0.6335:.0f} tok/s, 2 GB {best / 2:.0f} tok/s, 4 GB {best / 4:.1f} tok/s, 8 GB {best / 8:.1f} tok/s")
    except ImportError:
        print('numpy missing: pip install numpy for the bandwidth test')

    # the number that matters: what llama.cpp itself pulls through the memory system on a small dense model
    model = os.path.join(LAB, 'models', 'Qwen3-0.6B-Q8_0.gguf')
    if not os.path.exists(model): model = None
    exe = os.path.join(BIN, 'llama-bench.exe' if WIN else 'llama-bench')
    if model and os.path.exists(exe):
        t = '--threads' in sys.argv and int(sys.argv[sys.argv.index('--threads') + 1]) or (max(4, (n_phys or 8) // 2) if any('HYBRID' in w for w in warn) else (n_phys or n_log))
        r = subprocess.run([exe, '-m', model, '-p', '0', '-n', '64', '-r', '2', '-t', str(t), '-o', 'jsonl', '-v'], capture_output=True, text=True, timeout=300)
        dll = re.search(r'loaded CPU backend from .*?(ggml-cpu-[\w-]+\.(?:dll|so|dylib))', r.stderr + r.stdout)
        line = [l for l in r.stdout.splitlines() if l.startswith('{')]
        if line:
            j = json.loads(line[-1]); eff = j['model_size'] * j['avg_ts'] / 1e9
            print(f"llama-bench decode on {os.path.basename(model)} with -t {t}: {j['avg_ts']:.1f} tok/s = {eff:.1f} GB/s effective bandwidth (this is the figure to plan with; the streaming test above is a floor)")
            print(f"  ceilings at {eff:.1f} GB/s: 2 GB model {eff / 2:.0f} tok/s, 4 GB {eff / 4:.1f}, 8 GB {eff / 8:.1f}")
        if dll: print(f"  CPU kernels loaded: {dll.group(1)} (alderlake = AVX2 + AVX-VNNI; haswell = AVX2; sandybridge = AVX only; sse42/x64 = no AVX)")

    section('GPU')
    if shutil.which('nvidia-smi'):
        g = sh(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,clocks.max.sm', '--format=csv,noheader']).strip()
        print(g)
        pw = sh(['nvidia-smi', '-q', '-d', 'POWER'])
        cur = re.search(r'Current Power Limit\s*:\s*([\d.]+) W', pw); dft = re.search(r'Default Power Limit\s*:\s*([\d.]+) W', pw)
        if cur and dft:
            print(f"power limit now {cur.group(1)} W, default {dft.group(1)} W")
            if float(cur.group(1)) < 0.8 * float(dft.group(1)):
                warn.append(f'GPU POWER CAPPED at {cur.group(1)} W (default {dft.group(1)} W): battery or a quiet profile. Everything on the GPU runs at a fraction of its speed.')
        try:
            tot, used = [float(x.split()[0]) for x in g.split(',')[1:3]]
            print(f"free VRAM: {tot - used:.0f} MiB. Weights + n_ctx x KV bytes/token + ~10% compute must fit; on Windows an overflow spills to system RAM silently.")
        except Exception:
            pass
        apps = sh(['nvidia-smi', '--query-compute-apps=pid,name', '--format=csv,noheader']).strip()
        if apps: print('other CUDA processes:\n  ' + apps.replace('\n', '\n  '))
    else:
        print('no nvidia-smi. AMD: rocm-smi; Intel: xpu-smi; Apple silicon: the GPU shares the memory above, so the DIMM figure is the GPU figure too.')

    section('Background load')
    if WIN:
        load = ps("(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples[0].CookedValue").strip()
        try:
            l = float(load); print(f"CPU busy now: {l:.0f}%")
            if l > 10:
                top = ps("Get-Process | Sort-Object CPU -Descending | Select-Object -First 4 -ExpandProperty Name").split()
                warn.append(f"MACHINE BUSY ({l:.0f}% CPU; top by CPU time: {', '.join(top)}). Only paired, back-to-back comparisons will mean anything.")
        except ValueError:
            pass
    elif os.path.exists('/proc/loadavg'):
        print('loadavg:', open('/proc/loadavg').read().strip())

    if '--model' in sys.argv:
        path = sys.argv[sys.argv.index('--model') + 1]
        section(f'Model fit: {os.path.basename(path)}')
        size = os.path.getsize(path) / 2**20
        print(f"file {size:.0f} MiB; add n_ctx x KV bytes per token (zoo_meta.py prints it per model) and ~10% for compute buffers, then compare with the free figures above")

    section('Warnings')
    print('\n'.join(' - ' + w for w in warn) if warn else ' none')


if __name__ == '__main__':
    mp.freeze_support()
    main()

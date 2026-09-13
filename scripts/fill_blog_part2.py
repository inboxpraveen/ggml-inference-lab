"""Second half of the placeholder computation: speculative decoding, driver, load modes, ladder.
Imported by fill_blog.py after the bench/KLD placeholders are computed."""
import os, json


def compute(V, RES, ctx):
    rows, one, ts, speed, b_cpu_tg, b_gpu_tg = (ctx[k] for k in ['rows', 'one', 'ts', 'speed', 'b_cpu_tg', 'b_gpu_tg'])
    srv = [json.loads(l) for l in open(os.path.join(RES, 'server.jsonl'))]
    drv = [json.loads(l) for l in open(os.path.join(RES, 'driver.jsonl'))]

    def srv_get(tag, wl):
        rs = [r for r in srv if r['tag'] == tag and r['workload'] == wl]
        return rs[-1] if rs else None

    def drv_get(tag, loop, wl):
        rs = [r for r in drv if r['tag'] == tag and r['loop'] == loop and r['workload'] == wl]
        return rs[-1] if rs else None

    def cell(r, key='tok_s_mean', sd='tok_s_sd', d=1):
        return f"{r[key]:.{d}f}&nbsp;&plusmn;&nbsp;{r[sd]:.{d}f}" if r else '?'

    def acc(r):
        if not r: return ''
        if r.get('draft_n'): return f"{r['draft_n_accepted']} / {r['draft_n']} ({100*r['draft_n_accepted']/r['draft_n']:.0f}%)"
        if r.get('drafted'): return f"{r['accepted']} / {r['drafted']} ({100*r['accept_rate']:.0f}%)"
        return 'none'

    spec_rows = []

    def srow(label, tag, kind='srv', loop=None):
        if kind == 'srv':
            f, e, q = srv_get(tag, 'free'), srv_get(tag, 'edit'), srv_get(tag, 'summary_quote')
        else:
            f, e, q = drv_get(tag, loop, 'free'), drv_get(tag, loop, 'edit'), drv_get(tag, loop, 'summary_quote')
        spec_rows.append(f"                        <tr><td>{label}</td><td>{cell(f)}</td><td>{cell(e)}</td><td>{cell(q)}</td><td>{acc(e)}</td></tr>")

    spec_rows.append('                        <tr><td colspan="5"><strong>CPU, -t 8</strong></td></tr>')
    srow('llama-server, no speculation', 'srv-cpu-none')
    srow('llama-server, ngram-simple (defaults: n=12, m=48)', 'srv-cpu-ngram-simple')
    srow('llama-server, ngram-mod (defaults)', 'srv-cpu-ngram-mod')
    srow('llama-server, ngram-cache', 'srv-cpu-ngram-cache')
    srow('llama-server, ngram-simple, n=3, m=8', 'srv-cpu-ngram-simple-3-8')
    srow('driver, greedy loop', 'drv-cpu', 'drv', 'greedy_loop')
    srow('driver, n-gram loop, n=3, draft 8', 'drv-cpu', 'drv', 'ngram_spec_loop')
    srow('driver, n-gram loop, n=2, draft 8', 'drv-cpu-ng2', 'drv', 'ngram_spec_loop')
    srow('driver, n-gram loop, n=3, draft 16', 'drv-cpu-ng16', 'drv', 'ngram_spec_loop')
    spec_rows.append('                        <tr><td colspan="5"><strong>CUDA, all layers on the GPU</strong></td></tr>')
    srow('llama-server, no speculation', 'srv-cuda-none-rerun')
    srow('llama-server, ngram-simple (defaults)', 'srv-cuda-ngram-simple')
    srow('llama-server, ngram-cache', 'srv-cuda-ngram-cache')
    srow('llama-server, ngram-simple, n=3, m=16', 'srv-cuda-ngram-simple-3-16-rerun')
    srow('driver, greedy loop', 'drv-cuda', 'drv', 'greedy_loop')
    srow('driver, n-gram loop, n=3, draft 8', 'drv-cuda', 'drv', 'ngram_spec_loop')
    srow('driver, n-gram loop, n=3, draft 16', 'drv-cuda-ng16', 'drv', 'ngram_spec_loop')
    spec_rows.append('                        <tr><td colspan="5"><strong>Qwen3-8B Q6_K, -ngl 28, 4K context</strong></td></tr>')
    srow('llama-server, no speculation', 'srv-8b-none')
    srow('llama-server, ngram-mod (defaults)', 'srv-8b-ngram-mod')
    srow('llama-server, draft model = Qwen3-0.6B Q8_0, up to 6 tokens', 'srv-8b-draft-0.6b')
    V['SPEC_ROWS'] = '\n'.join(spec_rows)

    sn = srv_get('srv-cpu-none', 'edit'); s38 = srv_get('srv-cpu-ngram-simple-3-8', 'edit')
    s38q = srv_get('srv-cpu-ngram-simple-3-8', 'summary_quote'); snq = srv_get('srv-cpu-none', 'summary_quote')
    gs = srv_get('srv-cuda-none-rerun', 'edit'); g316 = srv_get('srv-cuda-ngram-simple-3-16-rerun', 'edit')
    dg = drv_get('drv-cpu', 'greedy_loop', 'edit'); dn = drv_get('drv-cpu', 'ngram_spec_loop', 'edit')
    dgg = drv_get('drv-cuda', 'greedy_loop', 'edit'); dng = drv_get('drv-cuda', 'ngram_spec_loop', 'edit')
    dng16 = drv_get('drv-cuda-ng16', 'ngram_spec_loop', 'edit')
    V['SPEC_COMMENTARY'] = (
        f"The defaults are the first lesson. llama-server's <code>ngram-simple</code> looks for a 12-token match and drafts 48 "
        f"tokens at a time, sizes chosen for long documents on big GPUs. On a 240-token editing prompt it found a match one time "
        f"in six and paid for the misses: {cell(srv_get('srv-cpu-ngram-simple','edit'))} tok/s against {cell(sn)} with no "
        f"speculation at all. <code>ngram-mod</code> drafted once in the whole run. <code>ngram-cache</code>, which keeps a "
        f"statistics table rather than a fixed match length, did well out of the box. And the same <code>ngram-simple</code> mode "
        f"with a 3-token match and 8-token drafts, the parameters my driver uses, went to {cell(s38)} tok/s on editing and "
        f"{cell(s38q)} on quoting, {s38['tok_s_mean']/sn['tok_s_mean']:.1f}x and {s38q['tok_s_mean']/snq['tok_s_mean']:.1f}x, with "
        f"no loss on free writing. Eighty tokens per second on a CPU whose single-token ceiling is 70 is the point of the exercise: "
        f"speculation is the only technique in this post that gets past the bandwidth wall, because it reads the weights once for "
        f"several tokens. On the GPU the same setting gave {cell(g316)} against {cell(gs)}, and my driver's loop landed in the "
        f"same place, {cell(dng)} with 8-token drafts and {cell(dng16)} with 16, because verifying sixteen tokens on a launch-bound "
        f"GPU costs about the same as verifying one. On the CPU the longer draft was worse, since sixteen extra tokens of real "
        f"arithmetic per step is no longer free there. Draft length should follow the device."
    )

    def drow(loop, label):
        cells = []
        for tag in ['drv-cpu', 'drv-cuda']:
            for wl in ['free', 'edit', 'summary_quote']:
                cells.append(f"<td>{cell(drv_get(tag, loop, wl))}</td>")
        return f"                        <tr><td>{label}</td>" + ''.join(cells) + "</tr>"
    V['DRIVER_ROWS'] = '\n'.join([drow('greedy_loop', 'Greedy, numpy argmax'), drow('sampler_loop', 'Greedy, llama.cpp sampler in C'),
                                  drow('ngram_spec_loop', 'n-gram draft (n=3, 8 tokens)')])
    sg = drv_get('drv-cpu', 'greedy_loop', 'free'); ss = drv_get('drv-cpu', 'sampler_loop', 'free'); sv = srv_get('srv-cpu-none', 'free')
    gg = drv_get('drv-cuda', 'greedy_loop', 'free'); gss = drv_get('drv-cuda', 'sampler_loop', 'free'); gsv = srv_get('srv-cuda-none-rerun', 'free')
    sge = drv_get('drv-cpu', 'greedy_loop', 'edit'); sve = srv_get('srv-cpu-none', 'edit')
    V['DRIVER_COMMENTARY'] = (
        f"The first two rows are the answer to the overhead question. Doing the argmax in Python over 151,936 floats versus doing "
        f"it in C changes nothing measurable: {cell(sg)} against {cell(ss)} tok/s on the CPU, {cell(gg)} against {cell(gss)} on "
        f"the GPU, both inside the run-to-run noise. A numpy argmax over that many floats takes about twenty microseconds on this "
        f"machine (I timed it); a decode step takes 27 milliseconds on the CPU and 3.5 on the GPU. The driver against llama-server "
        f"is a less tidy comparison, because the two were measured in different sessions on a machine whose load changes: on the "
        f"CPU the server was ahead on free writing, {cell(sv)} to {cell(sg)}, and behind on editing, {cell(sve)} to {cell(sge)}, "
        f"which is what noise looks like, not a gap. On the GPU the driver was consistently a little ahead, {cell(gg)} to "
        f"{cell(gsv)}, and my guess is the server's per-token work, sampling, streaming and bookkeeping, is a visible share of a "
        f"3.5 ms step where it isn't of a 27 ms one. Either way the C++ tools aren't leaving anything on the table that a custom "
        f"loop can pick up by being leaner. What the driver buys isn't speed; it's control. The n-gram loop with parameters tuned "
        f"to the workload is what gets the {dn['tok_s_mean']/dg['tok_s_mean']:.1f}x on the CPU and "
        f"{dng16['tok_s_mean']/dgg['tok_s_mean']:.1f}x on the GPU in the third row, and it took forty lines to write."
    )

    lm = json.load(open(os.path.join(RES, 'load_modes.json')))
    V['LOAD_MMAP'] = f"{lm['auto']['mean']:.2f}"; V['LOAD_NONE'] = f"{lm['none']['mean']:.2f}"

    lad = []
    def lrow(step, setting, r, what):
        lad.append(f"                        <tr><td>{step}</td><td><code>{setting}</code></td><td>{r}</td><td>{what}</td></tr>")
    ph = {n: one('paired-head-cpu-r1', 'tg', model_filename=f'models/variants/{n}.gguf')['avg_ts'] for n in ['plain-Q8_0', 'head-Q6_K-body-Q8_0']}
    lrow('CPU 1', '-t 16 (default) vs -t 8, same run, working load', f"{ts(b_cpu_tg)} vs {ts(one('pair-cpu-t8', 'tg'))}", 'threads = P-cores; 5 to 23% on an idle machine')
    lrow('CPU 2', 'Q8_0 vs Q8_0 body with Q6_K head, -t 8, quiet paired run', f"{ph['plain-Q8_0']:.1f} vs {ph['head-Q6_K-body-Q8_0']:.1f}", '6% fewer bytes, same accuracy')
    lrow('CPU 3', 'llama-server plain vs --spec-type ngram-simple n=3 m=8, editing task', f"{cell(sn)} vs {cell(s38)}", 'past the single-token bandwidth wall')
    lrow('CPU 3b', 'same, free writing', f"{cell(srv_get('srv-cpu-none', 'free'))} vs {cell(srv_get('srv-cpu-ngram-simple-3-8', 'free'))}", 'nothing to draft, nothing lost')
    lrow('GPU 1', 'llama-bench defaults vs -ngl 28 (head on CPU)', f"{ts(one('ngl-sweep', 'tg', n_gpu_layers=99))} vs {ts(one('ngl-sweep', 'tg', n_gpu_layers=28))}", 'keep the head on the card')
    lrow('GPU 2', '-fa on vs -fa off', f"{ts(one('fa-kv', 'tg', flash_attn=1))} vs {ts(one('fa-kv', 'tg', flash_attn=0))}", 'flash attention on')
    lrow('GPU 3', 'llama-server plain vs ngram-simple n=3 m=16, editing task', f"{cell(gs)} vs {cell(g316)}", 'drafts are nearly free when launch-bound')
    V['LADDER_ROWS'] = '\n'.join(lad)
    V['BASE_CPU_TG_ONLY'] = f"{b_cpu_tg['avg_ts']:.0f}"
    V['LADDER_THREADS_MULT'] = f"{one('pair-cpu-t8', 'tg')['avg_ts'] / b_cpu_tg['avg_ts']:.1f}"
    V['LADDER_HEAD_MULT'] = f"{ph['head-Q6_K-body-Q8_0'] / ph['plain-Q8_0']:.2f}"
    V['LADDER_SPEC_MULT'] = f"{s38['tok_s_mean']/sn['tok_s_mean']:.1f}"
    V['SPEC_8B_DRAFT'] = cell(srv_get('srv-8b-draft-0.6b', 'edit')); V['SPEC_8B_NONE'] = cell(srv_get('srv-8b-none', 'edit'))
    V['SPEC_8B_DRAFT_FREE'] = cell(srv_get('srv-8b-draft-0.6b', 'free')); V['SPEC_8B_NONE_FREE'] = cell(srv_get('srv-8b-none', 'free'))

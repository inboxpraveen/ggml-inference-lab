"""
Benchmark llama-server end to end over HTTP, including its built-in speculative decoding modes.

Starts one server per configuration, sends the same three workloads used by bench_driver.py with
temperature 0, and records the server's own timings (prompt_per_second, predicted_per_second) plus the
draft statistics it returns when speculation is on. Results go to results/server.jsonl.
"""
import argparse, json, os, subprocess, sys, time, urllib.request, statistics
sys.path.insert(0, os.path.dirname(__file__))
from bench_driver import PROMPTS

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PORT = 18089


def chat(user_text):
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def wait_health(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def completion(prompt, n_predict):
    body = json.dumps(dict(prompt=prompt, n_predict=n_predict, temperature=0, cache_prompt=False,
                           samplers=[], top_k=1)).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/completion', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', default='cuda')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n-predict', type=int, default=256)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--workloads', default='free,edit,summary_quote')
    ap.add_argument('server_args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    exe = os.path.join(LAB, 'bin', a.bin, 'llama-server.exe')
    args = [x for x in a.server_args if x != '--']
    cmd = [exe, '--host', '127.0.0.1', '--port', str(PORT), '--no-webui', '-c', '4096', '--prio', '2'] + args
    log = open(os.path.join(LAB, 'results', f'server-{a.tag}.log'), 'w')
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_health():
            print('server did not come up', a.tag); return
        for wl in a.workloads.split(','):
            prompt = chat(PROMPTS[wl])
            completion(prompt, 16)   # warm-up
            rows = []
            for _ in range(a.reps):
                rows.append(completion(prompt, a.n_predict))
            t = [r['timings'] for r in rows]
            tps = [x['predicted_per_second'] for x in t]
            rec = dict(tag=a.tag, backend=a.bin, workload=wl, args=' '.join(args),
                       n_prompt=t[0]['prompt_n'], n_gen=t[0]['predicted_n'],
                       tok_s_mean=statistics.mean(tps), tok_s_sd=statistics.pstdev(tps),
                       prompt_tok_s=statistics.mean(x['prompt_per_second'] for x in t),
                       draft_n=t[0].get('draft_n'), draft_n_accepted=t[0].get('draft_n_accepted'),
                       text_head=rows[0]['content'][:120])
            with open(os.path.join(LAB, 'results', 'server.jsonl'), 'a') as f:
                f.write(json.dumps(rec) + '\n')
            extra = ''
            if rec['draft_n']:
                extra = f"  draft {rec['draft_n_accepted']}/{rec['draft_n']} = {rec['draft_n_accepted']/rec['draft_n']:.2f}"
            print(f"{a.tag:28s} {wl:14s} {rec['tok_s_mean']:7.2f} ± {rec['tok_s_sd']:5.2f} tok/s  pp {rec['prompt_tok_s']:7.1f}{extra}")
    finally:
        p.terminate()
        try:
            p.wait(10)
        except Exception:
            p.kill()
        log.close()


if __name__ == '__main__':
    main()

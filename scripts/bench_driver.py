"""
End-to-end generation benchmark for the custom driver.

Compares, on the same llama.dll and the same model file:
  greedy   : Python loop, one llama_decode per token, numpy argmax on a zero-copy logits view
  sampler  : same loop, argmax done by llama.cpp's C sampler chain (isolates Python sampling cost)
  ngram    : self-speculative n-gram drafting (prompt lookup) with KV rollback

Two workloads: 'free' (open-ended writing, few n-gram hits) and 'edit' (return a corrected copy of a
document, many n-gram hits). Results are appended as JSON lines to results/driver.jsonl.
"""
import argparse, json, os, sys, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from driver import Engine

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DOC = """The llama.cpp project keeps a small number of CPU backend libraries in every Windows release.
Each library is compiled for one instruction set family, and the loader picks the best one at start-up by
asking every library to score itself against the processor it is running on. On an Intel Raptor Lake chip
the alderlake library wins because it is the only variant that pairs AVX2 with the AVX-VNNI dot product
instruction. If a variant's required feature bit is missing the score is zero and the library is skipped.
This is why copying a single library into the folder is a valid way to force a variant, and why forcing
an AVX-512 variant on this laptop produces no CPU backend at all. The thread count is a separate decision:
the default is the number of physical cores, which on a hybrid processor includes the efficiency cores,
and decode speed collapses when the barrier at the end of every matrix multiply has to wait for them."""

PROMPTS = {
    'free': "Write a detailed, technical explanation of why token generation in a large language model is "
            "limited by memory bandwidth rather than by arithmetic throughput. Cover the roofline model, "
            "the difference between prefill and decode, and what quantization changes.",
    'edit': "Here is a paragraph with a few typos. Return the corrected paragraph in full, changing nothing "
            "except the typos.\n\n" + DOC.replace('libraries', 'libraies', 1).replace('processor', 'proccessor', 1)
            .replace('separate', 'seperate', 1),
    'summary_quote': "Read the following text and then quote its third and fourth sentences verbatim, then "
            "explain them in one line each.\n\n" + DOC,
}


def run(engine, name, prompt_toks, n_predict, reps, **kw):
    rows = []
    for r in range(reps):
        fn = getattr(engine, name)
        res = fn(prompt_toks, n_predict, **kw)
        rows.append(res)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--bin', default=os.path.join(LAB, 'bin', 'cpu'))
    ap.add_argument('--ngl', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--n-predict', type=int, default=256)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--tag', default='')
    ap.add_argument('--workloads', default='free,edit,summary_quote')
    ap.add_argument('--n-gram', type=int, default=3)
    ap.add_argument('--n-draft', type=int, default=8)
    ap.add_argument('--loops', default='greedy_loop,sampler_loop,ngram_spec_loop')
    ap.add_argument('--print', action='store_true')
    a = ap.parse_args()

    eng = Engine(a.model, a.bin, n_gpu_layers=a.ngl, n_threads=a.threads, n_ctx=4096)
    out_path = os.path.join(LAB, 'results', 'driver.jsonl')
    for wl in a.workloads.split(','):
        toks = eng.tokenize(eng.apply_chat_template(PROMPTS[wl]), add_special=False, parse_special=True)
        ref = None
        for loop in a.loops.split(','):
            kw = {}
            if loop == 'ngram_spec_loop':
                kw = dict(n_gram=a.n_gram, n_draft=a.n_draft)
            # warm-up once (page cache, CUDA graphs), then measured reps
            getattr(eng, loop)(toks, 32, **kw)
            rows = run(eng, loop, toks, a.n_predict, a.reps, **kw)
            tps = [r['tok_s'] for r in rows]
            if ref is None:
                ref = rows[0]['tokens']
            same = all(r['tokens'] == ref for r in rows)
            rec = dict(tag=a.tag, backend=os.path.basename(a.bin), ngl=a.ngl, threads=a.threads,
                       model=os.path.basename(a.model), workload=wl, loop=loop, n_prompt=len(toks),
                       n_gen=rows[0]['n_gen'], tok_s_mean=statistics.mean(tps),
                       tok_s_sd=statistics.pstdev(tps) if len(tps) > 1 else 0.0,
                       prompt_tok_s=len(toks) / statistics.mean(r['prompt_s'] for r in rows),
                       identical_to_greedy=same, decode_calls=rows[0]['decode_calls'],
                       accept_rate=rows[0].get('accept_rate'), drafted=rows[0].get('drafted'),
                       accepted=rows[0].get('accepted'), n_gram=a.n_gram, n_draft=a.n_draft)
            with open(out_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            extra = f" accept={rec['accept_rate']:.2f} calls={rec['decode_calls']}" if rec['accept_rate'] is not None else ''
            print(f"{wl:14s} {loop:16s} {rec['tok_s_mean']:7.2f} ± {rec['tok_s_sd']:5.2f} tok/s  "
                  f"gen={rec['n_gen']} same={same}{extra}")
            if a.print:
                print('   ', repr(eng.detokenize(rows[0]['tokens'])[:300]))
    eng.close()


if __name__ == '__main__':
    main()

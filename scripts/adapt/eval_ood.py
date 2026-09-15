"""Score models on the out-of-distribution Hindi set (Hindi Wikipedia) with the same bits-per-byte code."""
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(__file__))
import fastattn; fastattn.install()
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import log_row
from eval_bpb import load_docs, bpb

ap = argparse.ArgumentParser()
ap.add_argument('models', nargs='+')
ap.add_argument('--tag', default=None)
ap.add_argument('--docs', type=int, default=400)
ap.add_argument('--chars', type=int, default=1500)
args = ap.parse_args()
docs = load_docs('hi_wiki.jsonl', args.docs, args.chars)
for path in args.models:
    tok = AutoTokenizer.from_pretrained(path); t = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    r = bpb(model, tok, docs, 4096)
    row = {'model': path, 'tag': args.tag or os.path.basename(path.rstrip('/')), 'params': sum(p.numel() for p in model.parameters()),
           'set': 'hi_wiki', 'docs': len(docs), 'chars_per_doc': args.chars, 'wiki': r, 'seconds': time.time() - t}
    print(f"{row['tag']:28s} wiki bpb {r['bpb']:.3f} (bpc {r['bpc']:.3f}, {r['tokens']} tok)  {row['seconds']:.0f}s")
    log_row('bpb_wiki', row)
    del model; torch.cuda.empty_cache()

"""SPECTRA-Flash vLLM plugin vs the decode engine (pod check, H100, vLLM 0.29). One step per process:

  ref   : the engine (spectra.flash.engine, flash8) greedy-decodes the test prompts, B = 1  -> <out>/ref.pt
  vllm  : vLLM with the plugin greedy-decodes the same prompts in one batch (--cg eager | full | default; --chunk N =
          chunked prefill in N-token steps, the approximation), plus the 32K needle                                                                              -> <out>/vllm_<cg>.pt
  force : the engine teacher-forces vLLM's tokens: argmax agreement and |delta logprob| of the chosen tokens (the
          meaningful metric: free-running greedy runs diverge at near-ties)
  cmp   : greedy token match of every saved vLLM run vs the engine
Prompts: a short chat prompt (crosses the 128-token exact window while decoding) and LongBench prompts.
vLLM needs the plugin entry point (spectra.flash.vllm:plugin, installed with the package) visible in every process.
usage: python tests/check_flash_vllm.py <step> --params <params dir or HF Hub repo> --out runs/flash_vllm_check [--cg eager]
"""
import argparse
import glob
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["ref", "vllm", "force", "cmp"])
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--params", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pick", default="qasper:0,multi_news:0,lcc:0,repobench-p:0", help="LongBench task:index prompts")
ap.add_argument("--tg", type=int, default=64)
ap.add_argument("--cg", default="eager", choices=["eager", "full", "default"])
ap.add_argument("--chunk", type=int, default=0, help="vllm: chunked prefill with this many tokens per step (0 = exact)")
ap.add_argument("--run", default="", help="force: the vLLM run file (default: vllm_eager.pt)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda")


def prompts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    msg = [{"role": "user", "content": "Write a short story about a lighthouse keeper who finds a message in a bottle."}]
    ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
    P = [(ids["input_ids"] if not torch.is_tensor(ids) else ids)[0]]
    from spectra.eval.longbench import encode, load_task
    for item in a.pick.split(","):
        task, i = item.split(":")
        P.append(encode(tok, task, load_task(task)[int(i)], 31500).long())
    return tok, P


def engine_decode(M, ids, forced=None):
    """Greedy (or teacher-forced) B = 1 decode; returns (tokens, logprob of each emitted / forced token, argmax)."""
    from spectra.flash import engine as EN
    E = EN.Engine(M, EN.Cache(M, "flash8", 1, ids.numel() + a.tg + 64))
    E.keep_logits = True
    E.prime()
    E.prefill(ids.to(dev))
    rows = [E.prefill_logits[0].float()]
    toks = []
    for s_ in range(a.tg):
        lg = rows[-1].log_softmax(-1)
        t = int(forced[s_]) if forced is not None else int(lg.argmax())
        toks.append((t, float(lg[t]), int(lg.argmax())))
        if s_ == a.tg - 1:
            break
        E.ids.fill_(t); E.set_decode(ids.numel() + s_)
        E.graph.replay()
        rows.append(E.logits[0].float().clone())
    del E
    torch.cuda.empty_cache()
    return toks


if a.step in ("ref", "force"):
    from spectra.flash import engine as EN
    tok, P = prompts()
    with torch.inference_mode():
        M = EN.Model(a.model, a.params, "cuda")
        if a.step == "ref":
            ref = [engine_decode(M, ids) for ids in P]
            torch.save(dict(prompts=P, ref=ref), os.path.join(a.out, "ref.pt"))
            for ids, r in zip(P, ref):
                print(f"[ref] {ids.numel():6d} tokens: {tok.decode([t for t, _, _ in r][:24])!r}", flush=True)
        else:
            run = torch.load(a.run or os.path.join(a.out, "vllm_eager.pt"))
            for i, ids in enumerate(P):
                vt = run["toks"][i]; vlp = run["logprobs"][i]
                f = engine_decode(M, ids, forced=vt)
                agree = sum(1 for t, _, am in f if am == t) / len(f)
                dlp = [abs(lp - v) for (_, lp, _), v in zip(f, vlp)]
                print(f"[force] prompt {i} ({ids.numel()} tokens): engine argmax = vLLM token {agree:.3f}, "
                      f"|dlogprob| mean {sum(dlp) / len(dlp):.4f} max {max(dlp):.4f}", flush=True)

elif a.step == "vllm":
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from spectra.flash import engine as EN
    tok, P = prompts()
    cc = {"cudagraph_mode": "FULL_DECODE_ONLY"} if a.cg == "full" else None
    llm = LLM(model=a.model, dtype="bfloat16", enable_prefix_caching=False, enable_chunked_prefill=a.chunk > 0,
              max_model_len=33024, max_num_batched_tokens=a.chunk or 33024, gpu_memory_utilization=0.85,
              enforce_eager=a.cg == "eager", compilation_config=cc, seed=0,
              hf_overrides={"architectures": ["SpectraFlashLlamaForCausalLM"], "spectra_flash": a.params})
    sp = SamplingParams(temperature=0.0, max_tokens=a.tg, ignore_eos=True, logprobs=1)
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids.tolist()) for ids in P], sp)
    toks, lps = [], []
    for o in outs:
        c = o.outputs[0]
        toks.append(list(c.token_ids))
        lps.append([lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)])
    nid, key = EN.needle_ids(tok, 32768)
    n_out = llm.generate([TokensPrompt(prompt_token_ids=nid.tolist())],
                         SamplingParams(temperature=0.0, max_tokens=12))[0].outputs[0].text
    print(f"[vllm {a.cg} chunk {a.chunk}] needle {nid.numel()} tokens: {'PASS' if key in n_out else 'FAIL'} ({n_out!r})", flush=True)
    for ids, t in zip(P, toks):
        print(f"[vllm {a.cg}] {ids.numel():6d} tokens: {tok.decode(t[:24])!r}", flush=True)
    tag = f"{a.cg}_chunk{a.chunk}" if a.chunk else a.cg
    torch.save(dict(toks=toks, logprobs=lps, needle=n_out), os.path.join(a.out, f"vllm_{tag}.pt"))

else:
    ref = torch.load(os.path.join(a.out, "ref.pt"))
    for f in sorted(glob.glob(os.path.join(a.out, "vllm_*.pt"))):
        run = torch.load(f)
        for i, (r, v) in enumerate(zip(ref["ref"], run["toks"])):
            rt = [t for t, _, _ in r]
            m = sum(x == y for x, y in zip(rt, v))
            fd = next((j for j, (x, y) in enumerate(zip(rt, v)) if x != y), None)
            print(f"[cmp] {os.path.basename(f):16s} prompt {i} ({ref['prompts'][i].numel()} tokens): {m}/{len(rt)} "
                  f"greedy tokens match the engine, first divergence {fd}", flush=True)
print("VLLM_CHECK_DONE", flush=True)

#!/usr/bin/env python3
"""SPECTRA-Flash on Qwen3.5 in vLLM 0.29 against dense vLLM: a pass key hidden in a ~28K-token document, and a short
chat. Each run is its own vLLM engine (a subprocess, one after the other on one GPU). Exits 1 if Flash misses the key.

    python tests/check_flash_qwen35.py --params morriszjm/Qwen3.5-9B-SPECTRA-Flash [--model Qwen/Qwen3.5-9B]
"""
import argparse
import json
import subprocess
import sys

RESULT = "CHECK_RESULT "
KEY = "48213"


def child(a) -> int:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=33024, max_num_batched_tokens=33024,
              enable_prefix_caching=False, enable_chunked_prefill=False, gpu_memory_utilization=0.85,
              language_model_only=True)
    hf = {}
    if a.variant == "flash":
        hf["spectra_flash"] = a.params
        kw["block_size"] = 540
    llm = LLM(hf_overrides=hf, **kw)
    tok = llm.get_tokenizer()

    def chat(text):
        x = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True,
                                    enable_thinking=False)
        return [int(i) for i in (x["input_ids"] if hasattr(x, "keys") else x)]
    fill = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    n = 28000 // len(tok(fill, add_special_tokens=False).input_ids)
    ids = [chat("Write a short story about a lighthouse keeper who finds a message in a bottle."),
           chat(fill * (n // 2) + f"The pass key is {KEY}. Remember it. {KEY} is the pass key. " + fill * (n - n // 2)
                + "\nWhat is the pass key? Answer with the number only.")]
    res = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], SamplingParams(max_tokens=64, temperature=0.0))
    print(RESULT + json.dumps(dict(variant=a.variant, lens=[len(x) for x in ids],
                                   texts=[r.outputs[0].text for r in res])), flush=True)
    return 0


def run(a, variant: str) -> dict:
    cmd = [sys.executable, __file__, "--child", "--variant", variant, "--model", a.model, "--params", a.params]
    p = subprocess.run(cmd, capture_output=True, text=True)
    res = next((json.loads(l[len(RESULT):]) for l in p.stdout.splitlines() if l.startswith(RESULT)), None)
    if p.returncode != 0 or res is None:
        sys.exit(f"{variant} run failed (exit {p.returncode}):\n" + "\n".join((p.stdout + p.stderr).splitlines()[-20:]))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--params", required=True, help="SPECTRA-Flash params dir or HF Hub repo id (Qwen3.5)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--variant", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        return child(a)
    out = {v: run(a, v) for v in ("dense", "flash")}
    for v, r in out.items():
        print(f"[{v}] chat:   {r['texts'][0][:120]!r}")
        print(f"[{v}] needle ({r['lens'][1]} tokens): {r['texts'][1].strip()!r} -> "
              f"{'PASS' if KEY in r['texts'][1] else 'FAIL'}")
    return 0 if KEY in out["flash"]["texts"][1] else 1


if __name__ == "__main__":
    sys.exit(main())

"""SPECTRA × vLLM demo — real int4-latent KV cache → ~7× KV-cache capacity at fixed GPU memory.

Two steps (two envs, on purpose):
  1) build the SPECTRA latent once with HF (transformers >=5.11) and save a tiny delta;
  2) serve it in vLLM (vllm 0.7.x + transformers 4.48.x) with the int4 latent cache.

    # step 1 (HF env):
    python demo/vllm_demo.py build --model NousResearch/Meta-Llama-3.1-8B-Instruct --out delta_vllm7x
    # step 2 (vLLM env):
    python demo/vllm_demo.py serve --model NousResearch/Meta-Llama-3.1-8B-Instruct --delta delta_vllm7x

`serve` prints the vLLM `num_gpu_blocks` (KV capacity) and the max concurrent sequences at several context
lengths vs the fp16 baseline — the headline memory result, measured end-to-end in a real serving engine.
"""
import argparse


def build(a):
    import spectra
    m, tok = spectra.compress(a.model, ratio=a.ratio, calib_windows=a.calib_windows)
    spectra.save(m, a.out)
    print(f"[demo] saved {a.ratio} delta ({m._spectra['compression']}x, r_k={m._spectra['r_k']}) -> {a.out}")


def serve(a):
    import spectra.vllm
    from vllm import SamplingParams
    llm = spectra.vllm.LLM(a.model, a.delta, gpu_memory_utilization=a.util, max_model_len=a.max_model_len)
    n = llm.llm_engine.cache_config.num_gpu_blocks
    cap = n * llm.llm_engine.cache_config.block_size
    print(f"[demo] SPECTRA int4-latent KV capacity: {n} blocks = {cap} tokens @ util={a.util}")
    for ctx in (4096, 8192, 16384, 32768):
        print(f"[demo]   ctx={ctx:>6}: max concurrent sequences = {cap // ctx}")
    out = llm.generate(["Explain in one sentence why the sky is blue.",
                        "The first president of the United States was"],
                       SamplingParams(max_tokens=32, temperature=0))
    for o in out:
        print("[gen]", repr(o.outputs[0].text[:120]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="HF: compress + save a vLLM-friendly SPECTRA latent delta")
    b.add_argument("--model", required=True)
    b.add_argument("--ratio", default="vllm-7x", help="paging-friendly symmetric LR preset (vllm-7x)")
    b.add_argument("--calib-windows", type=int, default=16)
    b.add_argument("--out", required=True)
    s = sub.add_parser("serve", help="vLLM: serve with the int4 latent cache")
    s.add_argument("--model", required=True)
    s.add_argument("--delta", required=True)
    s.add_argument("--util", type=float, default=0.85)
    s.add_argument("--max-model-len", type=int, default=32768)
    a = ap.parse_args()
    (build if a.cmd == "build" else serve)(a)


if __name__ == "__main__":
    main()

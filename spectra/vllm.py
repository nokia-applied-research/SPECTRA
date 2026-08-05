"""SPECTRA in vLLM — real low-bit latent KV cache (Mode-A).

Stores the SPECTRA **int4 latent** (r-dim, symmetric uniform-bit LR) directly in vLLM's paged KV cache, so
vLLM allocates ~N× more KV blocks at fixed GPU memory (real capacity / concurrency gain, measured
end-to-end). On read it gathers the latent by block-table, dequantizes, reconstructs K/V (up-projection),
applies EXACT RoPE, and attends. Reconstruction is training-free and exact (Mode A); a production fused
Triton decode kernel is future work (see README → Efficiency).

Only the paging-friendly symmetric uniform-bit LR points are supported here (fixed width, byte-aligned);
the water-filling / heavy-k presets stay HF-only (quality mode, `spectra.compress`).

Usage
-----
    import spectra, spectra.vllm
    # 1) build the SPECTRA latent once (HF, tf>=5.11) and save the tiny delta:
    m, tok = spectra.compress("NousResearch/Meta-Llama-3.1-8B-Instruct", ratio="vllm-7x")
    spectra.save(m, "spectra_l31_vllm7x")
    # 2) serve it in vLLM (int4 latent cache -> ~7x KV capacity):
    llm = spectra.vllm.LLM("NousResearch/Meta-Llama-3.1-8B-Instruct", "spectra_l31_vllm7x",
                           gpu_memory_utilization=0.85, max_model_len=32768)
    print(llm.generate(["The capital of France is"], ...))

Requires the vLLM stack (vllm 0.7.x + transformers 4.48.x); enforce_eager=True (the read path is a
Python reference — correct + memory-real, not yet the fused kernel).
"""
from __future__ import annotations
import os, math, torch


def _load_delta(delta):
    """delta = path to a `spectra.save` dir, or a list of per-layer tensor dicts."""
    if not isinstance(delta, str):
        return delta
    blob = torch.load(os.path.join(delta, "spectra.pt"), map_location="cpu", weights_only=False)
    out = []
    for spec in blob["layers"]:
        kt, vt = spec["k_tensors"], spec["v_tensors"]
        out.append(dict(Wdk=kt["W_DOWN"], Wuk=kt["W_UP"], muk=kt["mu"], betak=kt["beta"],
                        Wdv=vt["W_DOWN"], Wuv=vt["W_UP"], muv=vt["mu"], betav=vt["beta"],
                        oproj=spec["o_proj_weight"]))
    return out


def LLM(model_id, delta, *, group=64, gpu_memory_utilization=0.85, max_model_len=32768,
        enforce_eager=True, device="cuda:0", **llm_kwargs):
    """Return a vLLM `LLM` serving `model_id` with a SPECTRA int4 latent KV cache built from `delta`
    (a symmetric uniform-bit LR delta from `spectra.compress(ratio="vllm-7x")` + `spectra.save`)."""
    D = _load_delta(delta)
    R = int(D[0]["Wdk"].shape[1])                       # latent width per side (K==V, symmetric)
    if R % group != 0:
        raise ValueError(f"latent width R={R} must be divisible by group={group}")
    if R % 128 != 0:
        raise ValueError(f"latent width R={R} must be a multiple of 128 (pseudo-head packing)")
    NG = R // group
    BYTES = R // 2 + NG * 2 * 2                          # int4 codes + per-group (lo,scale) fp16
    LH = R // 128                                        # latent pseudo-heads (for the reshape)
    SP = [{k: d[k].to(device).to(torch.bfloat16)
           for k in ("Wdk", "Wuk", "muk", "betak", "Wdv", "Wuv", "muv", "betav")}
          | {"oproj": d["oproj"].to(device)} for d in D]

    # ---- int4 per-group affine (matches spectra.method.fake_quant: 16 levels, per-group min/max) ----
    def q_pack(c):                                       # c [T,R] -> [T,BYTES] uint8
        T = c.shape[0]; x = c.float().view(T, NG, group)
        lo = x.amin(-1, keepdim=True); hi = x.amax(-1, keepdim=True)
        sc = (hi - lo).clamp_min(1e-8) / 15
        q = ((x - lo) / sc).round().clamp(0, 15).to(torch.uint8).view(T, R)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
        lo16 = lo.squeeze(-1).half().contiguous().view(torch.uint8).view(T, NG * 2)
        sc16 = sc.squeeze(-1).half().contiguous().view(torch.uint8).view(T, NG * 2)
        return torch.cat([packed, lo16, sc16], 1)

    def unpack_dq(b):                                    # [N,BYTES] uint8 -> [N,R] bf16
        N = b.shape[0]; packed = b[:, :R // 2]
        lo = b[:, R // 2:R // 2 + NG * 2].contiguous().view(torch.float16).float().view(N, NG, 1)
        sc = b[:, R // 2 + NG * 2:].contiguous().view(torch.float16).float().view(N, NG, 1)
        q = torch.stack([(packed & 0xF), (packed >> 4)], -1).view(N, R).float().view(N, NG, group)
        return (q * sc + lo).view(N, R).to(torch.bfloat16)

    def rope_k(rope, Krec, positions):
        L = Krec.shape[0]; kf = Krec.reshape(L, -1)
        dummy = torch.zeros(L, 128, device=kf.device, dtype=kf.dtype)
        _, kr = rope.forward_native(positions, dummy, kf)
        return kr.view(L, 8, 128)

    # ---- vLLM CacheEngine: allocate a uint8 [2, blocks, block_size, BYTES] latent cache per layer ----
    from vllm.utils import get_dtype_size, LayerBlockType  # noqa: F401
    from vllm.worker.cache_engine import CacheEngine

    @staticmethod
    def _gcbs(cc, mc, pc):
        nl = mc.get_num_layers_by_block_type(pc, LayerBlockType.attention)
        return cc.block_size * BYTES * 2 * nl
    CacheEngine.get_cache_block_size = _gcbs

    def _alloc(self, num_blocks, dev):
        return [torch.zeros(2, num_blocks, self.block_size, BYTES, dtype=torch.uint8, device=dev)
                for _ in range(self.num_attention_layers)]
    CacheEngine._allocate_kv_cache = _alloc

    # ---- attention: write = quant+pack+scatter; read = gather+unpack+reconstruct+RoPE+flash ----
    import vllm.attention.backends.flash_attn as fa
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func as _flash
    except Exception:
        from flash_attn import flash_attn_varlen_func as _flash

    def _sp_fa(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
        sp = layer._sp; rope = layer._rope; pos = layer._pos
        if output is None:
            output = torch.empty_like(query)
        latK = key.reshape(key.shape[0], -1); latV = value.reshape(value.shape[0], -1)
        if kv_cache.numel() > 0:
            sm = attn_metadata.slot_mapping.flatten()
            kv_cache[0].view(-1, BYTES)[sm] = q_pack(latK)
            kv_cache[1].view(-1, BYTES)[sm] = q_pack(latV)
        scale = self.scale; npt = attn_metadata.num_prefill_tokens
        pm = attn_metadata.prefill_metadata; dm = attn_metadata.decode_metadata
        Wuk = sp["Wuk"]; bk = sp["betak"]; Wuv = sp["Wuv"]; bv = sp["betav"]
        if pm is not None and npt > 0:
            cK = unpack_dq(q_pack(latK[:npt])); cV = unpack_dq(q_pack(latV[:npt]))
            K = (cK @ Wuk + bk).view(npt, 8, 128); V = (cV @ Wuv + bv).view(npt, 8, 128)
            K = rope_k(rope, K, pos[:npt]); cu = pm.query_start_loc.to(torch.int32); ms = int(pm.max_prefill_seq_len)
            o = _flash(query[:npt], K.to(query.dtype), V.to(query.dtype), ms, cu, ms, cu,
                       softmax_scale=scale, causal=True)
            output[:npt] = o.to(output.dtype)
        if dm is not None:
            c0, c1 = kv_cache[0].view(-1, BYTES), kv_cache[1].view(-1, BYTES)
            bs = kv_cache[0].shape[1]; bt = dm.block_tables; sl = dm.seq_lens_tensor
            latK = []; latV = []; POS = []; cuk = [0]
            for s in range(bt.shape[0]):
                L = int(sl[s]); nb = (L + bs - 1) // bs; blk = bt[s, :nb]
                slots = (blk[:, None] * bs + torch.arange(bs, device=device)[None, :]).reshape(-1)[:L]
                latK.append(unpack_dq(c0[slots])); latV.append(unpack_dq(c1[slots]))
                POS.append(torch.arange(L, device=device)); cuk.append(cuk[-1] + L)
            LK = torch.cat(latK); LV = torch.cat(latV); P = torch.cat(POS)
            K = (LK @ Wuk + bk).view(-1, 8, 128); V = (LV @ Wuv + bv).view(-1, 8, 128); K = rope_k(rope, K, P)
            cuk = torch.tensor(cuk, device=device, dtype=torch.int32)
            cuq = torch.arange(bt.shape[0] + 1, device=device, dtype=torch.int32); msk = int(sl.max())
            o = _flash(query[npt:npt + bt.shape[0]], K.to(query.dtype), V.to(query.dtype), 1, cuq, msk, cuk,
                       softmax_scale=scale, causal=False)
            output[npt:npt + bt.shape[0]] = o.to(output.dtype)
        return output
    fa.FlashAttentionImpl.forward = _sp_fa

    import vllm.model_executor.models.llama as vlm

    def _sp_attn(self, positions, hidden_states, kv_cache, attn_metadata):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        x = hidden_states.to(torch.bfloat16); sp = self.attn._sp
        ck = (x @ sp["Wdk"] - sp["muk"]); cv = (x @ sp["Wdv"] - sp["muv"])   # continuous latent; quant in _sp_fa
        q, _ = self.rotary_emb(positions, q, k)
        self.attn._pos = positions; self.attn._rope = self.rotary_emb
        ao = self.attn(q, ck.to(q.dtype), cv.to(q.dtype), kv_cache, attn_metadata)
        out, _ = self.o_proj(ao)
        return out
    vlm.LlamaAttention.forward = _sp_attn

    _cnt = [0]; _orig_init = vlm.LlamaAttention.__init__

    def _sp_init(self, *a, **k):
        _orig_init(self, *a, **k)
        i = _cnt[0]; _cnt[0] += 1
        if i < len(SP):
            self.attn._sp = SP[i]; self.attn.num_kv_heads = LH; self.attn.impl.num_kv_heads = LH
            self.attn._pos = None; self.attn._rope = None
    vlm.LlamaAttention.__init__ = _sp_init

    from vllm import LLM as _VLLM
    llm = _VLLM(model=model_id, gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len, enforce_eager=enforce_eager, **llm_kwargs)
    layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    for i, lyr in enumerate(layers):
        lyr.self_attn.o_proj.weight.data.copy_(SP[i]["oproj"].to(lyr.self_attn.o_proj.weight.dtype))
    n = llm.llm_engine.cache_config.num_gpu_blocks
    print(f"[spectra.vllm] int4 latent cache R={R} bytes/token/side={BYTES} -> num_gpu_blocks={n} "
          f"(latent width {R} vs fp16 {8*128}={round(8*128*2/BYTES,2)}x/token)", flush=True)
    return llm

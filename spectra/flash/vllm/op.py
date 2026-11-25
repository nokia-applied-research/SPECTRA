"""`torch.ops.vllm.spectra_flash`: one SPECTRA-Flash layer's cache writes + attention, as one opaque op.

Input `qkvr` (T, NQKV + H r + R + r) = q | k | v | rope query | this layer's partial record entry (one GEMM in the
model code, as in the engine). For layer l at position li of its block:
  1. decode rows: the absorbed fp8 FlashMLA query (q_prep reads the un-roped q);
  2. RoPE on q and k of every token (vLLM's rotary kernel and table, in place);
  3. window write: each request's sinks and last `recent` tokens (+ the current one) into its ring (-1 = not kept);
  4. decode rows: FlashMLA over the record [sink, p - recent), the exact window, one softmax by LSE (tail_merge);
     prefill rows: a fresh prompt = exact causal attention over the prompt (FlashAttention varlen); a later chunk of
     a chunked prompt = the record + the ring as it was + the chunk itself (causal), merged by LSE (approximation);
  5. li == n - 1: the block's record entries of this step's tokens (positions >= sink), from the layers' partials.
The decode branch reads only persistent buffers and fixed shapes, so a decode step can be a FULL CUDA graph. The op is
a piecewise splitting op (added to CompilationConfig._attention_ops by `register`), so mixed batches run it eagerly.
"""
from __future__ import annotations

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

OP_NAME = "spectra_flash"
QUALIFIED_OP_NAME = f"vllm::{OP_NAME}"
_STEP: dict = {}            # FlashMLA tile metadata of the current step (computed by layer 0, reused by the others)
_K: dict = {}


def _kernels():
    if not _K:
        from vllm import _custom_ops
        from vllm.v1.attention.ops import flashmla
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        from .. import kernels
        _K.update(FK=kernels, MLA=flashmla, FA=flash_attn_varlen_func, OPS=_custom_ops)
    return _K["FK"], _K["MLA"], _K["FA"], _K["OPS"]


def _prefill_rows(rmd, nd: int, B: int) -> list:
    """[(request, first row, end row, cached tokens C)] of the prefill requests (C > 0: a continuation chunk)."""
    if nd >= B:
        return []
    qsl, seq = rmd.query_start_loc_cpu.tolist(), rmd.seq_lens_cpu.tolist()
    return [(b, qsl[b], qsl[b + 1], seq[b] - (qsl[b + 1] - qsl[b])) for b in range(nd, B) if qsl[b + 1] > qsl[b]]


def _ring_past(win, base, C: int, sink: int, ring: int):
    """K, V (n, HKV, D) of the exact keys a request's ring holds before position C: the sinks and the positions
    [max(sink, C - ring + 1), C). The slot that position C will take holds position C - ring (or older): excluded."""
    from ..hf import window_valid
    valid = window_valid(C - 1, sink, ring, base.device)
    if C - ring >= sink:
        valid[sink + C % ring] = False
    idx = base + torch.nonzero(valid).view(-1)
    return win.k[idx], win.v[idx]


def _chunk_attention(layer, FA, MLA, rec, rmd, b, q, k, v, qm, past, C, out):
    """One continuation chunk (S queries at positions C .. C + S - 1), the approximation of chunked prefill: the chunk
    sees the earlier tokens through the record [sink, C - recent) and the ring as it was at C (the sinks and the last
    recent tokens), plus itself causally. One softmax over both parts (merged by their LSE)."""
    S, H, HKV, D = q.shape[0], layer.num_heads, layer.num_kv_heads, layer.head_dim
    Kp, Vp = past
    P_ = Kp.shape[0]
    K = torch.cat([Kp, k.reshape(S, HKV, D)], 0)
    V = torch.cat([Vp, v.reshape(S, HKV, D)], 0)
    dev = q.device
    cu_q = torch.tensor([0, S], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, P_ + S], device=dev, dtype=torch.int32)
    o_e, lse_e = FA(q.reshape(S, H, D), K, V, max_seqlen_q=S, cu_seqlens_q=cu_q, max_seqlen_k=P_ + S,
                    cu_seqlens_k=cu_k, causal=True, softmax_scale=layer.scaling, fa_version=3, return_softmax_lse=True)
    lse_e = lse_e.reshape(H, S) if lse_e.shape[0] == H else lse_e.reshape(S, H).t()
    rl = C - layer.recent - layer.sink
    if rl <= 0:
        out.copy_(o_e.reshape(S, H * D))
        return
    rlen = torch.tensor([rl], device=dev, dtype=torch.int32)
    md, ns = MLA.get_mla_metadata_dense_fp8(rlen, S * H, 1)
    o_r, lse_r = MLA.flash_mla_with_kvcache_fp8(qm.view(1, S, H, layer.DK), rec.pages, rmd.block_table[b:b + 1], rlen,
                                                layer.R, md, ns, softmax_scale=layer.scaling, causal=False,
                                                descale_q=layer.qscale, descale_k=layer.one)
    o_r = torch.einsum("skgr,krd->skgd", o_r.view(S, HKV, layer.G, layer.R).float(), layer.WuV.float())
    w = torch.sigmoid(lse_r.view(H, S).float() - lse_e.float()).t()[:, :, None]       # record share, (S, H, 1)
    out.copy_((o_r.reshape(S, H, D) * w + o_e.float() * (1 - w)).reshape(S, H * D).to(out.dtype))


def run_layer(layer, qkvr: torch.Tensor, positions: torch.Tensor, output: torch.Tensor) -> None:
    md = get_forward_context().attn_metadata
    if isinstance(md, list):
        md = md[0]
    rec, win = layer.spf_record, layer.spf_window
    if not isinstance(md, dict) or rec.pages is None or win.k is None:
        output.zero_()                                          # profiling / dummy runs
        return
    FK, MLA, FA, OPS = _kernels()
    rmd, wmd = md[rec.prefix], md[win.prefix]
    T = qkvr.shape[0]
    H, HKV, D, DK = layer.num_heads, layer.num_kv_heads, layer.head_dim, layer.DK
    nd, B = rmd.num_decodes, rmd.num_reqs
    q = qkvr[:, :H * D]
    k = qkvr[:, H * D:(H + HKV) * D]
    v = qkvr[:, (H + HKV) * D:layer.NQKV]
    pre = _prefill_rows(rmd, nd, B)

    # 1. absorbed FlashMLA query of the decode rows and of continuation chunks with record entries (before RoPE)
    if nd > 0:
        qm = torch.empty(nd, H, DK, device=qkvr.device, dtype=torch.float8_e4m3fn)
        FK.q_prep(qkvr[:nd], layer.WuK, positions[:nd], layer.theta, layer.ss, layer.qscale, qm, True, H=H,
                  G=layer.G, D=D, QR0=layer.NQKV)
    qmc = {}
    for b, a0, e0, C in pre:
        if C - layer.recent - layer.sink > 0:
            qmc[b] = torch.empty(e0 - a0, H, DK, device=qkvr.device, dtype=torch.float8_e4m3fn)
            FK.q_prep(qkvr[a0:e0], layer.WuK, positions[a0:e0], layer.theta, layer.ss, layer.qscale, qmc[b], True,
                      H=H, G=layer.G, D=D, QR0=layer.NQKV)
    # 2. RoPE (in place on the q / k columns of qkvr)
    OPS.rotary_embedding(positions[:T], q, k, D, layer.rotary_emb.cos_sin_cache, True)
    # continuation chunks: the ring's exact keys as the chunk found them (read before this step's window writes)
    past = {b: _ring_past(win, wmd.wbase[b], C, layer.sink, layer.recent + 1) for b, a0, e0, C in pre if C > 0}
    # 3. exact window write
    OPS.reshape_and_cache_flash(k.view(T, HKV, D), v.view(T, HKV, D), win.k_cache, win.v_cache, wmd.wslot[:T], "auto",
                                layer.one, layer.one)
    # 4a. decode rows: record + window
    if nd > 0:
        if layer.spf_l == 0 or "md" not in _STEP:
            _STEP["md"], _STEP["ns"] = MLA.get_mla_metadata_dense_fp8(rmd.rlen_k[:nd], H, 1)
        o_l, lse = MLA.flash_mla_with_kvcache_fp8(qm.view(nd, 1, H, DK), rec.pages, rmd.block_table[:nd],
                                                  rmd.rlen_k[:nd], layer.R, _STEP["md"], _STEP["ns"],
                                                  softmax_scale=layer.scaling, causal=True, descale_q=layer.qscale,
                                                  descale_k=layer.one)
        FK.tail_merge(q[:nd], qkvr.stride(0), win.k, win.v, wmd.wbase[:nd], positions[:nd], rmd.rlen[:nd], o_l, lse,
                      layer.WuV, output[:nd], layer.scaling, layer.sink, layer.recent + 1, H=H, G=layer.G, D=D)
    # 4b. prefill rows: a fresh prompt = exact causal attention over itself; a continuation chunk = _chunk_attention
    end = nd
    for b, a0, e0, C in pre:
        S = e0 - a0
        if C == 0:
            cu = torch.tensor([0, S], device=qkvr.device, dtype=torch.int32)
            o = FA(q[a0:e0].view(S, H, D), k[a0:e0].view(S, HKV, D), v[a0:e0].view(S, HKV, D), max_seqlen_q=S,
                   cu_seqlens_q=cu, max_seqlen_k=S, cu_seqlens_k=cu, causal=True, softmax_scale=layer.scaling,
                   fa_version=3)
            output[a0:e0] = o.reshape(S, H * D)
        else:
            _chunk_attention(layer, FA, MLA, rec, rmd, b, q[a0:e0], k[a0:e0], v[a0:e0], qmc.get(b), past[b], C,
                             output[a0:e0])
        end = max(end, e0)
    if end < T:
        output[end:].zero_()                                    # padding tokens
    # 5. the block's record entries (layer li = 0 keeps its partial in a persistent buffer for li = 1)
    e = qkvr[:, layer.E0:]
    if layer.n == 1:
        FK.rec_write(e, None, positions[:T], layer.theta, rec.rows, rmd.rslot[:T], True, layer.R)
    elif layer.li == 0:
        layer.e0buf[:T].copy_(e)
    else:
        FK.rec_write(layer.e0buf[:T], e, positions[:T], layer.theta, rec.rows, rmd.rslot[:T], True, layer.R)


@eager_break_during_capture
def spectra_flash(qkvr: torch.Tensor, positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    run_layer(layer, qkvr, positions, output)


def spectra_flash_fake(qkvr: torch.Tensor, positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    return


_REGISTERED = False


def register_op() -> None:
    global _REGISTERED
    if _REGISTERED or hasattr(torch.ops.vllm, OP_NAME):
        _REGISTERED = True
        return
    direct_register_custom_op(op_name=OP_NAME, op_func=spectra_flash, mutates_args=["qkvr", "output"],
                              fake_impl=spectra_flash_fake)
    _REGISTERED = True


register_op()

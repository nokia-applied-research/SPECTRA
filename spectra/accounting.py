"""Compression accounting.

A preset's own ratio (`model._spectra["compression"]`) counts the latent payload: codes plus the fp16 scale and
zero of every group. The `-B` presets also keep a fixed window of exact fp16 tokens (sink 4 + recent 64 on both K
and V, w = 68), so their true ratio depends on the context length L:

    dense(L)   = L * 16
    spectra(L) = (L - min(L, w)) * b + min(L, w) * 16          b = 16 / latent_ratio
    ratio      = sum_i dense(L_i) / sum_i spectra(L_i)         over the samples actually evaluated

Every ratio in the README and the paper is this byte-weighted ratio over the evaluated samples.
"""


def honest_ratio(latent_ratio, lengths, window=68):
    """latent_ratio: the preset's latent-only ratio. lengths: context lengths (tokens) of the evaluated samples.
    window: exact fp16 tokens kept per sequence (sink + recent, 68 for the `-B` presets, 0 otherwise)."""
    if not lengths or not latent_ratio or window <= 0:
        return latent_ratio
    b = 16.0 / latent_ratio
    dense = sum(L * 16.0 for L in lengths)
    kept = sum((L - min(L, window)) * b + min(L, window) * 16.0 for L in lengths)
    return round(dense / kept, 4) if kept else latent_ratio


def accounting_block(latent_ratio, len_map, window=68):
    """Accounting record of a run: the ratio over all evaluated samples, plus the ratio at the median and at the
    maximum (31,500-token) context length. len_map: {task: [lengths]}."""
    all_lens = [L for lens in len_map.values() for L in lens]
    if not all_lens:
        return None
    med = sorted(all_lens)[len(all_lens) // 2]
    return {
        "window_tokens": window,
        "latent_ratio": latent_ratio,
        "ratio": honest_ratio(latent_ratio, all_lens, window),
        "ratio_at_median": honest_ratio(latent_ratio, [med], window),
        "ratio_at_maxlen": honest_ratio(latent_ratio, [31500], window),
        "n_samples": len(all_lens),
        "median_context_len": med,
        "definition": "sum_i dense(L_i) / sum_i spectra(L_i), byte-weighted over the evaluated samples",
    }


def flash_accounting(len_map, record_bytes=576, dense_bytes=4096, sink=4, window=127):
    """SPECTRA-Flash KV bytes per layer against dense, byte-weighted over the evaluated samples. Every token from
    position `sink` on has a fp8 record entry (512 latent + 64 RoPE dims = 576 B); the first `sink` and the last
    `window - sink` tokens also stay exact in bf16 (dense_bytes = 2 x KV width x 2 B; 4096 B on Llama-3.1-8B)."""
    all_lens = [L for lens in len_map.values() for L in lens]
    if not all_lens:
        return None
    dense = sum(L * dense_bytes for L in all_lens)
    flash = sum((L - min(L, sink)) * record_bytes + min(L, window) * dense_bytes for L in all_lens)
    return {"ratio": round(dense / flash, 4), "nominal_ratio": round(dense_bytes / record_bytes, 4),
            "record_bytes": record_bytes, "window_tokens": window, "n_samples": len(all_lens),
            "definition": "sum_i dense(L_i) / sum_i flash(L_i), record + exact window, byte-weighted"}

"""CPU checks (no model download): accounting, quantizers, exact window, presets.  python tests/test_cpu.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from spectra.accounting import honest_ratio, accounting_block
from spectra.method import fake_quant, LatentProjV2, cache_frac, _apply_exact_window
from spectra.quant import water_filling_bits
from spectra.api import PRESETS, NAMED, ALIASES, parse_preset, _default_args

ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name); ok &= bool(cond)

# accounting: no window -> latent ratio; window costs exactly 68 fp16 tokens per sequence
check("ratio without window", honest_ratio(12.19, [1000, 5000], window=0) == 12.19)
L = 10000; b = 16 / 12.19
check("ratio with window", abs(honest_ratio(12.19, [L]) - L * 16 / ((L - 68) * b + 68 * 16)) < 1e-3)
check("short sequence is all window", honest_ratio(12.19, [50]) == 1.0)
blk = accounting_block(12.19, {"a": [1000, 3000], "b": [31500]})
check("accounting block", blk["n_samples"] == 3 and blk["median_context_len"] == 3000)

# latent-only cache fraction of the 12x preset on an 8-KV-head model (kv width 1024, r_k 253, r_v 515)
a = _default_args(heavy_k=32, heavy_bits=8, group=64)
cf = cache_frac(253, 515, 3, a, 2048)
check(f"s12 latent ratio ({1 / cf:.2f}x)", 11.5 < 1 / cf < 13)

# water-filling: average bits preserved, high-variance groups get more bits
var = torch.tensor([8.0, 4.0, 2.0, 1.0, 0.01, 0.001])
bits = water_filling_bits(var, 3, 0, 8)
check(f"water-filling bits {bits.tolist()}", abs(float(bits.float().mean()) - 3) <= 0.5 and bits[0] >= bits[-1])

# fake quant: 8-bit error tiny, 2-bit error larger, shape kept
x = torch.randn(4, 7, 128)
e8 = (fake_quant(x, 8, 64) - x).abs().max().item(); e2 = (fake_quant(x, 2, 64) - x).abs().max().item()
check(f"fake_quant 8b err {e8:.4f} < 2b err {e2:.4f}", e8 < 0.05 and e2 > e8)

# exact window: first 4 and last 64 prompt positions equal the exact projection; middle quantized; decode untouched
torch.manual_seed(0)
d, r, m = 64, 16, 32
W = torch.randn(m, d) * 0.1
mod = LatentProjV2(torch.randn(d, r) * 0.1, torch.randn(r, m) * 0.1, torch.zeros(r), torch.zeros(m), 2, 16, 0, 8)
h = torch.randn(2, 100, d)
q = mod(h)
mod.sink, mod.recent, mod._W_exact, mod._b_exact = 4, 64, W.t(), None
o = mod(h); ex = h @ W.t()
check("window: sink exact", torch.allclose(o[:, :4], ex[:, :4]))
check("window: recent exact", torch.allclose(o[:, -64:], ex[:, -64:]))
check("window: middle quantized", torch.allclose(o[:, 4:36], q[:, 4:36]))
o1 = mod(h[:, :1]); mod.sink = mod.recent = 0; q1 = mod(h[:, :1])
check("window: decode token quantized", torch.equal(o1, q1) and not torch.allclose(o1, ex[:, :1]))

# presets
check("v0.1 names resolve", all(parse_preset(a)[0] == ALIASES[a] for a in ALIASES))
check("ratio names resolve", all(parse_preset(n)[0] + ("-B" if parse_preset(n)[2] else "") == c for n, c in NAMED.items()))
check("-B suffix turns the window on", parse_preset("s12-B")[2] and not parse_preset("s12")[2])
check("all presets complete", all({"mode", "bits", "kv_ratio", "group"} <= set(c) for c in PRESETS.values()))

print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

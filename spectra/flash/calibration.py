"""Calibration windows for the SPECTRA-Flash conversion: WikiText-2 train and Python source, cut into windows of
`seq` tokens (half text, half code)."""
from __future__ import annotations

# Code corpus candidates, tried in order (the-stack-smol is gated on the Hub). The released SPECTRA-Flash params
# use codeparrot-clean-valid (config.FLASH_CODE_SOURCES).
CODE_SOURCES = (
    ("bigcode/the-stack-smol", dict(data_dir="data/python", split="train"), "content", 800),
    ("codeparrot/codeparrot-clean-valid", dict(split="train"), "content", 800),
)


def calibration_windows(tok, seq=2048, nwin=32, code_frac=0.5, log=print, sources=CODE_SOURCES):
    from datasets import load_dataset
    nl = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
    code, src = "", None
    for name, kw, col, k in sources:
        try:
            ds = load_dataset(name, **kw)
            code = "\n\n".join(str(x) for x in ds[:k][col]); src = name
            break
        except Exception as e:                                           # gated / offline: try the next source
            log(f"[spectra-flash] code corpus {name} unavailable ({type(e).__name__}); trying the next one")
    nl_ids = tok(nl, return_tensors="pt").input_ids[0]
    code_ids = tok(code, return_tensors="pt").input_ids[0] if code else nl_ids
    nc = int(nwin * code_frac); nn = nwin - nc
    wins = [nl_ids[i * seq:(i + 1) * seq] for i in range(nn)] + [code_ids[i * seq:(i + 1) * seq] for i in range(nc)]
    wins = [w for w in wins if w.numel() == seq]
    return wins, {"text": "Salesforce/wikitext wikitext-2-raw-v1 train", "code": src or "none (text only)",
                  "windows": len(wins), "seq": seq}

#!/usr/bin/env python3
"""CLI entrypoint for the Needle-In-A-Haystack evaluation (heatmaps + baselines).
Runs without install: adds the repo root to sys.path, then calls spectra.eval.niah.main.

Example:
  python scripts/run_niah.py --model NousResearch/Meta-Llama-3.1-8B-Instruct \
      --contexts 1024,2048,4096,8192,16384,32768 --depths 0,17,33,50,67,83,100 \
      --center --heavy-k 16 --heavy-bits 8 --hadamard --configs 1024:2 \
      --baselines turboquant:2,kivi:4,snapkv:512 --out results/niah_l31.json \
      --fig-prefix results/figures/niah_l31
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectra.eval.niah import main

if __name__ == "__main__":
    main()

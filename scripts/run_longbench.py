#!/usr/bin/env python3
"""LongBench-8 for the dense model or a SPECTRA preset (official prompts and scorer, full test split).

  python scripts/run_longbench.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --out runs/dense.json
  python scripts/run_longbench.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --preset s95-B --out runs/s95B.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectra.eval.longbench import main

if __name__ == "__main__":
    main()

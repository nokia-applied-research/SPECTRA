#!/usr/bin/env python3
"""Needle-in-a-haystack grid (contexts 1K-32K x depths 0-100%) for the dense model or a SPECTRA preset.

  python scripts/run_niah.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --preset s12-B --out runs/niah.json
  python scripts/run_niah.py --model NousResearch/Meta-Llama-3.1-8B-Instruct --preset s12-B --needles 4 --out runs/niah4.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectra.eval.niah import main

if __name__ == "__main__":
    main()

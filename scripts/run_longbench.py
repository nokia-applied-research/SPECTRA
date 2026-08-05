#!/usr/bin/env python3
"""CLI entrypoint for LongBench evaluation. Runs without install (adds repo root to sys.path)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectra.eval.longbench import main

if __name__ == "__main__":
    main()

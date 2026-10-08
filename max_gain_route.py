"""The max-gain route finder now lives in the vertmaxxer package (github.com/mikegrudic/vertmaxxer).

This module stands in for it, so ``import max_gain_route`` and ``python max_gain_route.py ...`` keep working, with
this repo's download cache.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("VERTMAXXER_CACHE", str(Path(__file__).resolve().parent / "cache" / "max_gain_route"))
import vertmaxxer.core  # noqa: E402

if __name__ == "__main__":
    from vertmaxxer.cli import main

    main()
else:
    sys.modules[__name__] = vertmaxxer.core

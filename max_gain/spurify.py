"""Stands in for vertmaxxer-spurify (the vertmaxxer package), with this repo's download cache."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import max_gain_route  # noqa: E402,F401  (sets the cache)
from vertmaxxer.cli import spurify_main  # noqa: E402

if __name__ == "__main__":
    spurify_main()

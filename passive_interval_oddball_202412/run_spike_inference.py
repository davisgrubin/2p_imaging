#!/usr/bin/env python3
"""Thin wrapper around the shared spike inference CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from spike_inference.cli import main as shared_main

    shared_main()


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import warnings

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.warn(
    "train_diffusion_forecaster_WV.py is now a thin compatibility wrapper; use `neuralop.flood.cli.train_diffusion` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralop.flood.cli.train_diffusion import main

if __name__ == "__main__":
    raise SystemExit(main())

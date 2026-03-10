"""Thin compatibility facade for flood diffusion training."""

from __future__ import annotations

from neuralop.flood.train.diffusion_app import main

if __name__ == "__main__":
    raise SystemExit(main())

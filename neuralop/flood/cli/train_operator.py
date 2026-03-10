"""Compatibility facade for WV flood operator training helpers."""

from __future__ import annotations

from neuralop.flood.data.hec_ras import *
from neuralop.flood.data.datasets_impl import *
from neuralop.flood.data.normalization_impl import *
from neuralop.flood.processing.wv_impl import *
from neuralop.flood.train.debug import *
from neuralop.flood.train.fgn import *
from neuralop.flood.train.gaussian import *
from neuralop.flood.train.rollout import *
from neuralop.flood.train.operator_app import main
from neuralop.flood.utils.runtime_core import *
from neuralop.flood.visualization.publication import *

if __name__ == "__main__":
    raise SystemExit(main())

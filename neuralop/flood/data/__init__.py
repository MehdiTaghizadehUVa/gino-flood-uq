"""Flood dataset and HEC-RAS access interfaces."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "FloodDatasetHDF",
    "FloodRolloutTestDatasetHDF",
    "NormalizedDatasetOnTheFly",
    "NormalizedRolloutTestDataset",
    "build_cell_point_index",
    "collect_all_fields",
    "fit_normalizers_streaming",
    "get_hec_ras_hdf_shape",
    "read_hec_ras_hdf_slice",
    "read_hec_ras_hdf_static",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module("neuralop.flood.data.wv")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


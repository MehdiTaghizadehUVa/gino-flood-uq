"""Flood dataset and HEC-RAS access interfaces."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "FloodDatasetHDF",
    "FloodRolloutTestDatasetHDF",
    "NormalizedDatasetOnTheFly",
    "NormalizedRolloutTestDataset",
    "build_normalizer_metadata",
    "build_cell_point_index",
    "collect_all_fields",
    "fit_normalizers",
    "fit_normalizers_fast_exact",
    "fit_normalizers_streaming",
    "get_hec_ras_hdf_shape",
    "load_normalizer_metadata",
    "normalizer_metadata_matches",
    "read_hec_ras_hdf_slice",
    "read_hec_ras_hdf_static",
    "resolve_normalizer_fit_method",
    "resolve_normalizer_metadata_path",
    "save_normalizer_metadata",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module("neuralop.flood.data.wv")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

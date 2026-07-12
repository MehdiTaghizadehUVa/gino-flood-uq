import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    sys.modules.setdefault(name, types.ModuleType(name))


def _load_module(name: str, rel_path: str):
    _ensure_pkg("neuralop")
    _ensure_pkg("neuralop.flood")
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon = _load_module("neuralop.flood.neon", "neuralop/flood/neon.py")

epistemic_member_bootstrap_weights = neon.epistemic_member_bootstrap_weights
per_epistemic_fair_crps = neon.per_epistemic_fair_crps


def test_member_bootstrap_rows_sum_to_one_and_are_deterministic():
    z_e = torch.randn(5, 4)
    w1 = epistemic_member_bootstrap_weights(["fam-a", "fam-b"], z_e, 6, seed=11)
    w2 = epistemic_member_bootstrap_weights(["fam-a", "fam-b"], z_e, 6, seed=11)

    assert w1.shape == (2, 5, 6)
    torch.testing.assert_close(w1.sum(dim=-1), torch.ones(2, 5))
    torch.testing.assert_close(w1, w2)
    assert not torch.allclose(w1[:, 0], w1[:, 1])


def test_member_bootstrap_temperature_zero_is_uniform():
    z_e = torch.randn(3, 2)
    weights = epistemic_member_bootstrap_weights(["fam-a"], z_e, 4, temperature=0.0)
    expected = torch.full((1, 3, 4), 0.25)
    torch.testing.assert_close(weights, expected)


def test_member_bootstrap_is_keyed_by_original_member_ids():
    z_e = torch.randn(4, 3)
    full = epistemic_member_bootstrap_weights(["fam-a"], z_e, 6, seed=3)
    idx = torch.tensor([4, 1, 5])
    sub = epistemic_member_bootstrap_weights(
        ["fam-a"],
        z_e,
        3,
        member_indices=idx,
        seed=3,
    )
    sliced = full.index_select(-1, idx)
    sliced = sliced / sliced.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(sub, sliced)


def test_uniform_member_weights_match_unweighted_crps():
    pred = torch.tensor(
        [[[[[[0.0]]], [[[2.0]]]], [[[[4.0]]], [[[6.0]]]]]]
    )  # [B=1,M=2,K=2,T=1,Nv=1,C=1]
    ref = torch.tensor([[[[[0.0]]], [[[2.0]]], [[[4.0]]]]])
    uniform = torch.full((1, 2, 3), 1.0 / 3.0)

    weighted = per_epistemic_fair_crps(pred, ref, member_weights=uniform, reduction="none")
    unweighted = per_epistemic_fair_crps(pred, ref, reduction="none")

    torch.testing.assert_close(weighted, unweighted)


def test_weighted_crps_matches_hand_computed_two_member_case():
    pred = torch.tensor([[[[[[1.0]]], [[[3.0]]]]]])  # [B=1,M=1,K=2,T=1,Nv=1,C=1]
    ref = torch.tensor([[[[[0.0]]], [[[4.0]]]]])
    weights = torch.tensor([[[0.25, 0.75]]])

    actual = per_epistemic_fair_crps(pred, ref, member_weights=weights, reduction="none")
    # term1 = mean_k sum_r w_r |x_k-y_r| = (0.25*1 + 0.75*3 + 0.25*3 + 0.75*1)/2 = 2
    # fair term2 = 1/(2*K*(K-1)) sum_{k!=k'} |x_k-x_k'| = 1
    expected = torch.tensor([[1.0]])
    torch.testing.assert_close(actual, expected)

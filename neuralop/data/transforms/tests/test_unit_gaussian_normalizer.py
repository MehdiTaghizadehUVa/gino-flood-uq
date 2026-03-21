from ..normalizers import UnitGaussianNormalizer
import torch
from torch.testing import assert_close
from flaky import flaky

@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_created_from_stats(eps=1e-6):
    x = torch.rand(16, 3, 40, 50, 60)*2.5
    mean = torch.mean(x, dim=[0, 2, 3, 4], keepdim=True)
    std = torch.std(x, dim=[0, 2, 3, 4], keepdim=True)

    # Init normalizer with ground-truth mean and std
    normalizer = UnitGaussianNormalizer(mean=mean, std=std, eps=eps)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps

@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_from_data(eps=1e-6):
    x = torch.rand(16, 3, 40, 50, 60)*2.5
    mean = torch.mean(x, dim=[0, 2, 3, 4], keepdim=True)
    std = torch.std(x, dim=[0, 2, 3, 4], keepdim=True)   
    # Init by fitting whole data at once
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4], eps=eps)
    normalizer.fit(x)
    
    assert_close(normalizer.mean, mean)
    assert_close(normalizer.std, std, rtol=eps, atol=eps)

    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps
    
    assert_close(normalizer.mean, mean)
    assert_close(normalizer.std, std, rtol=eps, atol=eps)

@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_incremental_update(eps=1e-6):
    x = torch.rand(16, 3, 40, 50, 60)*2.5
    mean = torch.mean(x, dim=[0, 2, 3, 4], keepdim=True)
    std = torch.std(x, dim=[0, 2, 3, 4], keepdim=True)   
    # Incrementally compute mean and var
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4], eps=eps)
    normalizer.partial_fit(x, batch_size=2)

    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    assert_close(x_unnormalized, x)
    assert torch.mean(x_normalized) <= eps
    assert (torch.std(x_normalized) - 1) <= eps

    assert_close(normalizer.mean, mean)
    assert_close(normalizer.std, std, rtol=eps, atol=eps)

def test_UnitGaussianNormalizer_incremental_update_large_magnitude_geometry():
    n_cells = 2000
    chunk_batch = 50
    n_chunks = 20
    coords = torch.stack(
        [
            torch.linspace(300000.0, 301000.0, n_cells),
            torch.linspace(4300000.0, 4300800.0, n_cells),
        ],
        dim=1,
    ).float()
    chunks = [coords.unsqueeze(0).repeat(chunk_batch, 1, 1) for _ in range(n_chunks)]

    normalizer = UnitGaussianNormalizer(dim=[0, 1], eps=1e-6)
    normalizer.fit(chunks[0])
    for chunk in chunks[1:]:
        normalizer.partial_fit(chunk, batch_size=chunk.shape[0])

    all_data = torch.cat(chunks, dim=0)
    expected_mean = torch.mean(all_data, dim=[0, 1], keepdim=True)
    expected_std = torch.std(all_data, dim=[0, 1], keepdim=True)

    assert_close(normalizer.mean, expected_mean, rtol=1e-6, atol=1e-4)
    assert_close(normalizer.std, expected_std, rtol=1e-6, atol=1e-3)


import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# only import open3d if built
open3d_built = False
try:
    from open3d.ml.torch.layers import FixedRadiusSearch

    open3d_built = True
except:
    pass


def _build_neighbor_row_splits(nbrhd_sizes: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build CSR row splits, avoiding unsupported CUDA deterministic cumsum kernels."""
    cumsum_input = nbrhd_sizes
    if nbrhd_sizes.is_cuda and torch.are_deterministic_algorithms_enabled():
        cumsum_input = nbrhd_sizes.cpu()

    nbrhd_sizes_cumsum = torch.cumsum(cumsum_input, dim=0)
    splits = torch.cat(
        (torch.zeros(1, device=nbrhd_sizes_cumsum.device, dtype=nbrhd_sizes_cumsum.dtype), nbrhd_sizes_cumsum),
        dim=0,
    )
    if splits.device != device:
        splits = splits.to(device=device)
    return splits


def native_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
    """
    Native PyTorch implementation of a neighborhood search
    between two arbitrary coordinate meshes.

    Parameters
    -----------
    data : torch.Tensor
        Vector of data points from which to find neighbors. Shape: (num_data_points, dimensions)
    queries : torch.Tensor
        Centers of neighborhoods. Shape: (num_queries, dimensions)
    radius : float
        Size of each neighborhood

    Returns
    --------
    nbr_dict : dict
        A dictionary containing:
            - 'neighbors_index': Tensor of neighbor indices
            - 'neighbors_row_splits': Tensor indicating the start and end indices for each query's neighbors
            - 'average_num_neighbors': Scalar tensor representing the average number of neighbors per query
            - 'neighbor_counts': Tensor of shape (num_queries,) with the number of neighbors for each query
    """
    # Compute pairwise distances between queries and data points
    dists = torch.cdist(queries, data)  # (num_queries, num_data_points)

    # Create a binary mask where True indicates the data point is within the radius of the query
    in_nbr = dists <= radius  # (num_queries, num_data_points)

    # Extract the indices of data points that are neighbors to any query
    nbr_indices = in_nbr.nonzero(as_tuple=False)[:, 1]  # (total_num_neighbors,)

    # Compute the number of neighbors for each query
    nbrhd_sizes = torch.sum(in_nbr, dim=1, dtype=torch.long)  # (num_queries,)

    # Compute row splits in a deterministic-safe way when CUDA determinism is enabled.
    splits = _build_neighbor_row_splits(nbrhd_sizes, device=queries.device)

    # Calculate the average number of neighbors per query
    total_neighbors = torch.sum(nbrhd_sizes)
    num_queries = queries.shape[0]
    if num_queries == 0:
        average_num_neighbors = torch.tensor(0.0, device=queries.device)
    else:
        average_num_neighbors = total_neighbors.to(torch.float32) / num_queries

    # Prepare the neighborhood dictionary
    nbr_dict = {
        'neighbors_index': nbr_indices.long(),
        'neighbors_row_splits': splits.long(),
        'average_num_neighbors': average_num_neighbors,
        'neighbor_counts': nbrhd_sizes.long()  # number of neighbors for each query
    }

    return nbr_dict


def plot_query_nodes_with_neighbor_counts(queries, neighbor_counts):
    """
    Plot the query nodes (assumed to be 2D coordinates) and highlight nodes that have zero neighbors.

    Parameters:
    -----------
    queries : torch.Tensor or numpy.ndarray
        Query coordinates of shape (num_queries, 2).
    neighbor_counts : torch.Tensor or numpy.ndarray
        Number of neighbors for each query, of shape (num_queries,).
    """
    # Convert tensors to NumPy arrays if necessary.
    if isinstance(queries, torch.Tensor):
        queries = queries.detach().cpu().numpy()
    if isinstance(neighbor_counts, torch.Tensor):
        neighbor_counts = neighbor_counts.detach().cpu().numpy()

    # Create a mask: True for queries with zero neighbors.
    zero_mask = (neighbor_counts == 0)

    plt.figure(figsize=(8, 6))
    # Plot queries with neighbors in blue.
    plt.scatter(queries[~zero_mask, 0], queries[~zero_mask, 1], c='blue', s=5, label='With neighbors')
    # Plot queries with zero neighbors in red.
    plt.scatter(queries[zero_mask, 0], queries[zero_mask, 1], c='red', s=5, label='Zero neighbors')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.title('Query nodes (Red: Zero neighbors, Blue: With neighbors)')
    plt.legend()
    plt.show()


class NeighborSearch(nn.Module):
    """
    Neighborhood search between two arbitrary coordinate sets.
    For each point `x` in `queries`, returns a set of the indices of all points `y` in `data`
    within the ball of radius r `B_r(x)`.

    Parameters
    ----------
    use_open3d : bool
        Whether to use Open3D or native PyTorch implementation.
        NOTE: Open3D implementation requires 3D data.
    """

    def __init__(self, use_open3d=True):
        super().__init__()
        if use_open3d and open3d_built:  # slightly faster, works on GPU in 3D only
            self.search_fn = FixedRadiusSearch()
            self.use_open3d = True
        else:  # fallback: works on GPU and CPU for any dimension
            self.search_fn = native_neighbor_search
            self.use_open3d = False

    def forward(self, data, queries, radius, compute_norm=None):
        """
        Find the neighbors in `data` of each point in `queries` within a ball of `radius`.
        Returns the result in Compressed Row Storage (CRS) format.

        Parameters
        ----------
        data : torch.Tensor of shape [n, d]
            Search space of possible neighbors (NOTE: Open3D requires d=3).
        queries : torch.Tensor of shape [m, d]
            Points for which to find neighbors (NOTE: Open3D requires d=3).
        radius : float
            Radius of each ball: B(queries[j], radius).
        compute_norm : None or bool, default None
            If not None (or True), compute squared L2 norms for each neighbor pair,
            stored under return_dict['norm'].
        debug : bool, default False
            If True, print out neighbor counts and plot the query nodes with zero neighbors.

        Returns
        ----------
        return_dict : dict
            A dictionary with at least the following keys:
                - 'neighbors_index': torch.Tensor (dtype=torch.int64)
                - 'neighbors_row_splits': torch.Tensor (shape [m+1], dtype=torch.int64)
                - 'neighbor_counts': torch.Tensor (shape [m], dtype=torch.int64)
                - 'average_num_neighbors': scalar tensor
            If compute_norm is True, also contains:
                - 'norm': torch.Tensor of shape [total_num_neighbors]
                        (squared distances between neighbor pairs).
        """
        # Use Open3D's FixedRadiusSearch if possible
        if self.use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict = {
                'neighbors_index': search_return.neighbors_index.long(),
                'neighbors_row_splits': search_return.neighbors_row_splits.long()
            }
        else:
            return_dict = self.search_fn(data, queries, radius)

        # Optionally compute squared norms for each neighbor pair
        if compute_norm is not None:
            nbr_splits = return_dict['neighbors_row_splits']
            num_reps = nbr_splits[1:] - nbr_splits[:-1]
            rep_queries = torch.repeat_interleave(queries, num_reps, dim=0)
            rep_data = data[return_dict['neighbors_index']]
            rep_dist = rep_queries - rep_data
            return_dict['norm'] = (rep_dist ** 2).sum(dim=-1)

        return return_dict

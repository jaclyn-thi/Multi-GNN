import torch

def build_adj(edge_index, num_nodes):
    """
    Builds adjacency matrix for a (batched) graph.

    Args:
        edge_index: (2, E)
        num_nodes: int

    Returns:
        A: (N, N) boolean adjacency matrix
    """
    device = edge_index.device
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)

    src, dst = edge_index
    A[src, dst] = True

    return A


def get_adj_from_data(data):
    from data_loading import get_forward_edge_index

    edge_index = get_forward_edge_index(data)
    return build_adj(edge_index, data.x.size(0))

import torch
from data_loading import GraphData

def random_edge_drop(edge_index, edge_attr, drop_rate):
    """
    Randomly drops edges from a directed multigraph.

    Args:
        edge_index: (2, E)
        edge_attr: (E, D)
        drop_rate: float in [0,1]

    Returns:
        new_edge_index, new_edge_attr
    """
    E = edge_index.size(1)

    if E == 0:
        return edge_index, edge_attr

    keep_mask = torch.rand(E, device=edge_index.device) > drop_rate

    # ensure at least one edge remains
    if keep_mask.sum() == 0:
        rand_idx = torch.randint(0, E, (1,), device=edge_index.device)
        keep_mask[rand_idx] = True

    return edge_index[:, keep_mask], edge_attr[keep_mask]


def make_view(data, edge_index, edge_attr):
    view = data.clone()
    view.edge_index = edge_index
    view.edge_attr = edge_attr
    return view


def generate_views(data, drop_rate):
    view1 = data.clone()
    view2 = data.clone()
    return view1, view2


# def generate_views(data, drop_rate):
#     """
#     Returns two edge-dropped views of the SAME graph.
#     Preserves GraphData type + all attributes.
#     """

#     ei1, ea1 = random_edge_drop(data.edge_index, data.edge_attr, drop_rate)
#     ei2, ea2 = random_edge_drop(data.edge_index, data.edge_attr, drop_rate)

#     view1 = data.clone()
#     view1.edge_index = ei1
#     view1.edge_attr = ea1

#     view2 = data.clone()
#     view2.edge_index = ei2
#     view2.edge_attr = ea2

#     return view1, view2

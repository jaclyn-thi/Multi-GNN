import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import OptTensor
import numpy as np

def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1) if not isinstance(data, HeteroData) else torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out

def to_adj_edges_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    # calculate adjacent edges with times per node
    adj_edges_out = dict([(i, []) for i in range(num_nodes)])
    adj_edges_in = dict([(i, []) for i in range(num_nodes)])
    for i, (u,v,t) in enumerate(edges):
        u,v,t = int(u), int(v), int(t)
        adj_edges_out[u] += [(i, v, t)]
        adj_edges_in[v] += [(i, u, t)]
    return adj_edges_in, adj_edges_out

def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports

def time_deltas(data, adj_edges_list):
    time_deltas = torch.zeros(data.edge_index.shape[1], 1)
    if data.timestamps is None:
        return time_deltas
    for v, edges in adj_edges_list.items():
        if len(edges) < 1: continue
        a = np.array(edges)
        a = a[a[:, -1].argsort()]
        a_tds = [0] + [a[i+1,-1] - a[i,-1] for i in range(a.shape[0]-1)]
        tds = np.hstack((a[:,0].reshape(-1,1), np.array(a_tds).reshape(-1,1)))
        for i,td in tds:
            time_deltas[i] = td
    return time_deltas

class GraphData(Data):
    '''This is the homogenous graph object we use for GNN training if reverse MP is not enabled'''
    def __init__(
        self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, 
        readout: str = 'edge', 
        num_nodes: int = None,
        timestamps: OptTensor = None,
        node_timestamps: OptTensor = None,
        **kwargs
        ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        self.loss_fn = 'ce'
        self.num_nodes = int(self.x.shape[0])
        self.node_timestamps = node_timestamps
        if timestamps is not None:
            self.timestamps = timestamps  
        elif edge_attr is not None:
            self.timestamps = edge_attr[:,0].clone()
        else:
            self.timestamps = None

    def add_ports(self):
        '''Adds port numberings to the edge features'''
        reverse_ports = True
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self.edge_index, adj_list_in)
        out_ports = [ports(self.edge_index.flipud(), adj_list_out)] if reverse_ports else []
        self.edge_attr = torch.cat([self.edge_attr, in_ports] + out_ports, dim=1)
        return self

    def add_time_deltas(self):
        '''Adds time deltas (i.e. the time between subsequent transactions) to the edge features'''
        reverse_tds = True
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = [time_deltas(self, adj_list_out)] if reverse_tds else []
        self.edge_attr = torch.cat([self.edge_attr, in_tds] + out_tds, dim=1)
        return self

class HeteroGraphData(HeteroData):
    '''This is the heterogenous graph object we use for GNN training if reverse MP is enabled'''
    def __init__(
        self,
        readout: str = 'edge',
        **kwargs
        ):
        super().__init__(**kwargs)
        self.readout = readout

    @property
    def num_nodes(self):
        return self['node'].x.shape[0]
        
    @property
    def timestamps(self):
        return self['node', 'to', 'node'].timestamps

    def add_ports(self):
        '''Adds port numberings to the edge features'''
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self['node', 'to', 'node'].edge_index, adj_list_in)
        out_ports = ports(self['node', 'rev_to', 'node'].edge_index, adj_list_out)
        self['node', 'to', 'node'].edge_attr = torch.cat([self['node', 'to', 'node'].edge_attr, in_ports], dim=1)
        self['node', 'rev_to', 'node'].edge_attr = torch.cat([self['node', 'rev_to', 'node'].edge_attr, out_ports], dim=1)
        return self

    def add_time_deltas(self):
        '''Adds time deltas (i.e. the time between subsequent transactions) to the edge features'''
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = time_deltas(self, adj_list_out)
        self['node', 'to', 'node'].edge_attr = torch.cat([self['node', 'to', 'node'].edge_attr, in_tds], dim=1)
        self['node', 'rev_to', 'node'].edge_attr = torch.cat([self['node', 'rev_to', 'node'].edge_attr, out_tds], dim=1)
        return self
    
def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std


def resolve_directional_edge_feature_schema(edge_dim: int, *, ports: bool, tds: bool) -> dict:
    """Resolve named edge-feature columns from construction order (not trailing indices).

    Homogeneous AML construction in ``data_loading.get_data`` appends:
      base edge features → [in_port, out_port] if ports → [in_td, out_td] if tds

    Both port and TDS pairs are source/destination directional: under reverse MP the
    destination becomes the source and vice versa, so each pair must swap once on the
    reverse relation when corrected semantics are enabled.
    """
    n_extra = (2 if ports else 0) + (2 if tds else 0)
    if edge_dim < n_extra:
        raise ValueError(
            f"edge_dim={edge_dim} too small for ports={ports} tds={tds} (need >= {n_extra})"
        )
    base_dim = int(edge_dim) - n_extra
    names = [f"base_{i}" for i in range(base_dim)]
    indices: dict = {}
    col = base_dim
    swap_pairs = []
    if ports:
        names.extend(["in_port", "out_port"])
        indices["in_port"] = col
        indices["out_port"] = col + 1
        swap_pairs.append(("in_port", "out_port", col, col + 1))
        col += 2
    if tds:
        names.extend(["in_td", "out_td"])
        indices["in_td"] = col
        indices["out_td"] = col + 1
        swap_pairs.append(("in_td", "out_td", col, col + 1))
        col += 2
    return {
        "edge_dim": int(edge_dim),
        "base_dim": base_dim,
        "ports": bool(ports),
        "tds": bool(tds),
        "names": names,
        "indices": indices,
        "swap_pairs": swap_pairs,
    }


def apply_corrected_reverse_edge_attr(edge_attr: torch.Tensor, *, ports: bool, tds: bool):
    """Clone ``edge_attr`` and swap directional source/dest pairs for reverse MP.

    Does not mutate ``edge_attr``. Nondirectional (base) columns are unchanged.
    """
    schema = resolve_directional_edge_feature_schema(
        int(edge_attr.shape[1]), ports=ports, tds=tds
    )
    rev = edge_attr.clone()
    for _name_a, _name_b, i, j in schema["swap_pairs"]:
        # Swap exactly once per directional pair.
        rev[:, [i, j]] = rev[:, [j, i]]
    return rev, schema


def create_hetero_obj(x,  y,  edge_index,  edge_attr, timestamps, args):
    '''Creates a heterogenous graph object for reverse message passing'''
    data = HeteroGraphData()

    data['node'].x = x
    data['node', 'to', 'node'].edge_index = edge_index
    data['node', 'rev_to', 'node'].edge_index = edge_index.flipud()
    ports = bool(getattr(args, "ports", False))
    tds = bool(getattr(args, "tds", False))
    correct = bool(getattr(args, "correct_reverse_edge_features", False))

    if correct:
        # Independent reverse storage; forward stays in original orientation.
        data['node', 'to', 'node'].edge_attr = edge_attr
        rev_attr, schema = apply_corrected_reverse_edge_attr(
            edge_attr, ports=ports, tds=tds
        )
        data['node', 'rev_to', 'node'].edge_attr = rev_attr
        semantics = "corrected"
    else:
        # Inherited upstream behavior (default / paper Multi-GIN+EU path):
        # forward and reverse alias the same storage; if ports=True, an in-place
        # swap of the trailing two columns mutates both relations. With ports+TDS
        # those trailing columns are TDS, not ports.
        data['node', 'to', 'node'].edge_attr = edge_attr
        data['node', 'rev_to', 'node'].edge_attr = edge_attr
        if ports:
            #swap the in- and outgoing port numberings for the reverse edges
            data['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]] = data['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]
        schema = resolve_directional_edge_feature_schema(
            int(edge_attr.shape[1]), ports=ports, tds=tds
        )
        semantics = "inherited_legacy"

    data['node', 'to', 'node'].y = y
    data['node', 'to', 'node'].timestamps = timestamps
    # Diagnostics only (HeteroData allows custom attributes).
    data.reverse_edge_feature_semantics = semantics
    data.edge_feature_schema = schema
    data.correct_reverse_edge_features = correct

    return data


RGCN_REL_FORWARD = 0.0
RGCN_REL_REVERSE = 1.0


def rgcn_num_relations(reverse_mp: bool) -> int:
    return 2 if reverse_mp else 1


def _append_relation_column(edge_attr: torch.Tensor, relation_id: float) -> torch.Tensor:
    col = torch.full(
        (edge_attr.shape[0], 1),
        relation_id,
        dtype=edge_attr.dtype,
        device=edge_attr.device,
    )
    return torch.cat([edge_attr, col], dim=1)


def append_rgcn_relation_type(data, reverse_mp: bool):
    """Append relation ids as the final edge feature column (not z-normalized)."""
    if reverse_mp:
        data['node', 'to', 'node'].edge_attr = _append_relation_column(
            data['node', 'to', 'node'].edge_attr,
            RGCN_REL_FORWARD,
        )
        data['node', 'rev_to', 'node'].edge_attr = _append_relation_column(
            data['node', 'rev_to', 'node'].edge_attr,
            RGCN_REL_REVERSE,
        )
    else:
        data.edge_attr = _append_relation_column(data.edge_attr, RGCN_REL_FORWARD)
    return data
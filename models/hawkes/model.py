import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import Linear
from torch_geometric.utils import softmax


def simple_coalesce(edge_index: Tensor, num_nodes: int | None = None, sort_by_row: bool = True):
    nnz = edge_index.size(1)
    key = edge_index.new_empty(nnz + 1)
    key[0] = -1
    key[1:] = edge_index[1 - int(sort_by_row)]
    key[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])
    key_sorted, perm = key[1:].sort()
    e_sorted = edge_index[:, perm]
    mask = key_sorted > key[:-1]
    uniq = e_sorted if mask.all() else e_sorted[:, mask]
    idx_map = torch.arange(0, nnz, device=edge_index.device)
    idx_map.sub_(mask.logical_not_().cumsum(dim=0))
    return e_sorted, uniq, perm, idx_map


def binary_deg_snorm(edge_index: Tensor, unique_edge_index: Tensor, num_nodes: int) -> Tensor:
    row_u, col_u = unique_edge_index
    ones = torch.ones(unique_edge_index.size(1), device=edge_index.device)
    deg = scatter_add(ones, col_u, dim=0, dim_size=num_nodes)  # in-degree
    deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
    row, col = edge_index
    return deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E]


class HGCNConv(MessagePassing):
    """
    H^{k+1} = D^{-1/2} C D^{-1/2} H^{k} W,  with  C_e = exp(-delta_src * dt_e)
    Full-batch only. Degree norm uses binary adjacency.
    """

    def __init__(self, n_node: int, in_channels: int, out_channels: int, dropout: float = 0.0, bias: bool = False):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.decay = Parameter(torch.ones(n_node, 1))  # per-source-node theta should be ≥ 0
        # self._decay_raw = Parameter(torch.full((n_node, 1), -2.0))

        self.dropout = dropout
        self.bias = Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: Tensor, edge_index: Tensor, edge_age: Tensor) -> Tensor:
        dt = edge_age.view(-1, 1).clamp_min(0.0)  # [E,1]
        dec = self.decay[edge_index[0]].clamp_min(0.0)  # [E,1]
        # dec = F.softplus(self._decay_raw)[edge_index[0]]
        C = torch.exp(-dt * dec).squeeze(-1)  # [E]

        if self.training and self.dropout > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropout
            edge_index = edge_index[:, mask]
            C = C[mask] / (1.0 - self.dropout)

        _, uniq, _, _ = simple_coalesce(edge_index, num_nodes=x.size(0))
        snorm = binary_deg_snorm(edge_index, uniq, x.size(0))  # [E]
        weight = (C * snorm).view(-1, 1)

        h = self.lin(x)
        out = self.propagate(edge_index, x=h, edge_weight=weight)
        if self.bias is not None: out = out + self.bias
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight * x_j


class HawkesGATConv(MessagePassing):
    """
    Full-batch Hawkes-GAT without edge features in attention.
    alpha_ij = softmax_j(LeakyReLU(a_src·x_i + a_dst·x_j))
    weight_ij = alpha_ij * exp(-alpha_ij * dt_ij)
    out_i = sum_j weight_ij * (W x_j)
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 2,
                 dropout: float = 0.0, negative_slope: float = 0.2,
                 concat: bool = True, bias: bool = False):
        super().__init__(aggr='add', node_dim=0)
        self.heads, self.out_channels, self.concat = heads, out_channels, concat
        self.dropout, self.negative_slope = dropout, negative_slope

        self.lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

        self.bias = Parameter(torch.zeros(heads * out_channels if concat else out_channels)) if bias else None

    def forward(self, x: Tensor, edge_index: Tensor, edge_age: Tensor) -> Tensor:
        H, C = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, C)  # [N,H,C]

        src, dst = edge_index
        x_j, x_i = x[src], x[dst]  # [E,H,C]

        e = (x_i * self.att_src).sum(-1) + (x_j * self.att_dst).sum(-1)  # [E,H]
        e = F.leaky_relu(e, self.negative_slope)
        alpha = softmax(e, dst, num_nodes=x.size(0))  # [E,H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        dt = edge_age.view(-1, 1).clamp_min(0.0)  # [E,1]
        weight = alpha * torch.exp(-alpha * dt)  # [E,H]

        # out = self.propagate(edge_index, x_j=x_j, weight=weight,size=(x.size(0), x.size(0)))  # [N,H,C]
        out = self.propagate(edge_index, x=x, weight=weight, size=(x.size(0), x.size(0)))  # [N,H,C]
        out = out.view(-1, H * C) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, weight: Tensor) -> Tensor:
        return weight.unsqueeze(-1) * x_j


class NodeHawkGNN(nn.Module):
    """Binary node classification with Hawkes-{GCN,GAT} (no edge feats in conv)."""

    def __init__(self, gnn_type: str, n_node: int, in_dim: int, hid_dim: int,
                 layers: int = 2, dropout: float = 0.1, use_bn: bool = True,
                 heads: int = 2, concat: bool = True):
        super().__init__()
        assert gnn_type in {"GCN", "GAT"}
        self.gnn_type = gnn_type
        self.input = nn.Linear(in_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None

        if gnn_type == "GCN":
            for _ in range(layers):
                self.convs.append(HGCNConv(n_node, hid_dim, hid_dim, dropout=dropout))
                if self.bns is not None: self.bns.append(nn.BatchNorm1d(hid_dim))
            out_dim = hid_dim
        else:
            for _ in range(layers):
                self.convs.append(HawkesGATConv(
                    in_channels=hid_dim,
                    out_channels=(hid_dim // heads) if concat else hid_dim,
                    heads=heads, dropout=dropout, concat=concat,
                ))
                out_dim = (hid_dim // heads) * heads if concat else hid_dim
                if self.bns is not None: self.bns.append(nn.BatchNorm1d(out_dim))
        self.cls = nn.Linear(out_dim, 1)
        self.postprocessing_anomaly = nn.Linear(out_dim, 1)

    def forward(self, x: Tensor, edge_index: Tensor, edge_age: Tensor):
        h = self.input(x)
        for l, conv in enumerate(self.convs):
            if self.gnn_type == "GCN":
                h = conv(h, edge_index, edge_age)
            else:
                h = conv(h, edge_index, edge_age)  # no edge_attr
            if self.bns is not None: h = self.bns[l](h)
            if l < len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        logits = self.cls(h).squeeze(-1)  # [N] binary logit
        anomaly_score = self.postprocessing_anomaly(h).squeeze(-1)
        return logits, anomaly_score, h


class EdgeHawkGNN(nn.Module):
    """
    Binary edge classification with Hawkes-{GCN,GAT}.
    - GNN does NOT use edge features.
    - Final decoder concatenates [z_src, z_dst, edge_features_for_decoder].
    """

    def __init__(self, gnn_type: str, n_node: int, in_dim: int, hid_dim: int,
                 layers: int = 2, dropout: float = 0.1, use_bn: bool = True,
                 heads: int = 2, concat: bool = True,
                 edge_attr_dim: int = 0):
        super().__init__()
        assert gnn_type in {"GCN", "GAT"}
        self.gnn_type = gnn_type
        self.input = nn.Linear(in_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None

        if gnn_type == "GCN":
            for _ in range(layers):
                self.convs.append(HGCNConv(n_node, hid_dim, hid_dim, dropout=dropout))
                if self.bns is not None: self.bns.append(nn.BatchNorm1d(hid_dim))
            out_dim = hid_dim
        else:
            for _ in range(layers):
                self.convs.append(HawkesGATConv(
                    in_channels=hid_dim,
                    out_channels=(hid_dim // heads) if concat else hid_dim,
                    heads=heads, dropout=dropout, concat=concat,
                ))
                out_dim = (hid_dim // heads) * heads if concat else hid_dim
                if self.bns is not None: self.bns.append(nn.BatchNorm1d(out_dim))

        self.cls = nn.Linear(2 * out_dim + edge_attr_dim, 1)
        self.postprocessing_anomaly = nn.Linear(2 * out_dim + edge_attr_dim, 1)

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor, edge_attr,
                edge_age: Tensor):
        h = self.input(x)
        for l, conv in enumerate(self.convs):
            if self.gnn_type == "GCN":
                h = conv(h, edge_index, edge_age)
            else:
                h = conv(h, edge_index, edge_age)  # no edge_attr
            if self.bns is not None: h = self.bns[l](h)
            if l < len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        z_src, z_dst = h[edge_label_index[0]], h[edge_label_index[1]]
        feat = torch.cat([z_src, z_dst, edge_attr], dim=1)
        logits = self.cls(feat).view(-1)  # [#edges_to_score]
        anomaly_score = self.postprocessing_anomaly(feat).squeeze(-1)
        return logits, anomaly_score, h

import torch_geometric.nn as geom_nn
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch.nn import Linear, GRU


class NodeGNN(nn.Module):
    def __init__(self, input_dim, hidden_size=16, out_put_size=2, gnn_type="GCN",
                 num_layers=2,
                 dropout=0.0, heads=1):
        super().__init__()
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, hidden_size)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == "GAT":
                self.convs.append(pyg_nn.GATConv(hidden_size, hidden_size // heads, heads=heads))
            elif gnn_type == "GIN":
                self.convs.append(pyg_nn.GINConv(
                    nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())))
            else:  # GCN
                self.convs.append(pyg_nn.GCNConv(hidden_size, hidden_size))
        self.postprocessing1 = geom_nn.Linear(hidden_size, out_put_size)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.postprocessing1.reset_parameters()

    def forward(self, x, edge_index):
        # Preprocess step
        # h is embedding, out classification, anomaly_score is for anomaly score
        h = self.preprocess1(x)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.postprocessing1(h)
        out = torch.sum(out, dim=-1)
        return out, h

    def get_embedding(self, x, edge_index):
        _, embeddings = self.forward(x, edge_index)
        return embeddings


class EdgeGNN(nn.Module):
    def __init__(self, input_dim, edge_attr_dim=0, hidden_size=16, gnn_type="GCN",
                 num_layers=2,
                 dropout=0.0, heads=1):
        super().__init__()
        self.preprocess1 = Linear(input_dim, 128)
        self.preprocess2 = Linear(128, hidden_size)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == "GAT":
                self.convs.append(pyg_nn.GATConv(hidden_size, hidden_size // heads, heads=heads))
            elif gnn_type == "GIN":
                self.convs.append(pyg_nn.GINConv(
                    nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())))
            else:  # GCN
                self.convs.append(pyg_nn.GCNConv(hidden_size, hidden_size))
        self.postprocessing1 = geom_nn.Linear(2 * hidden_size + edge_attr_dim, 1)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocessing1.reset_parameters()

    def forward(self, x, edge_index, edge_label_index, edge_attr):
        # Preprocess step
        # h is embedding, out classification, anomaly_score is for anomaly score
        h = self.preprocess1(x)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        combined = torch.cat([h_src, h_dst, edge_attr], dim=1)
        out = self.postprocessing1(combined)
        # h = self.postprocessing1(h_hat)
        # CLS layer
        out = out.view(-1)
        return out, h

    def get_embedding(self, x, edge_index):
        _, embeddings = self.forward(x, edge_index)
        return embeddings

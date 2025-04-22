import torch_geometric.nn as geom_nn
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch.nn import Linear, GRU


# gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}


class NodeDyFraudNet(nn.Module):
    def __init__(self, input_dim, memory_size=16, hidden_size=16, out_put_size=2, enable_memory=True, gnn_type="GCN",
                 num_layers=2,
                 dropout=0.0, heads=1):
        super().__init__()
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, hidden_size)
        if enable_memory:
            self.conv1 = EvolveGNN_O(hidden_size, memory_size, hidden_size, gnn_type)
            self.conv2 = EvolveGNN_O(hidden_size, memory_size, hidden_size, gnn_type)
        else:
            if gnn_type == "GAT":
                self.conv1 = pyg_nn.GATConv(hidden_size, hidden_size // heads, heads=heads)
                self.conv2 = pyg_nn.GATConv(hidden_size, hidden_size // heads, heads=heads)
            elif gnn_type == "GIN":
                self.conv1 = pyg_nn.GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
                self.conv2 = pyg_nn.GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
            else:  # GCN
                self.conv1 = pyg_nn.GCNConv(hidden_size, hidden_size)
                self.conv2 = pyg_nn.GCNConv(hidden_size, hidden_size)
        self.postprocessing1 = geom_nn.Linear(hidden_size, out_put_size)
        self.postprocessing_anomaly = geom_nn.Linear(hidden_size, 1)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocessing1.reset_parameters()

    def forward(self, x, edge_index):
        # Preprocess step
        # h_hat is embedding, h classification, h_anomaly is for anomaly score
        h = self.preprocess1(x)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.conv1(h, edge_index)

        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h, inplace=False)
        h_hat = F.dropout(h, p=self.dropout, inplace=True)
        h = self.postprocessing1(h_hat)
        h = torch.sum(h, dim=-1)
        anomaly_score = self.postprocessing_anomaly(h_hat).squeeze(-1)
        return h, anomaly_score, h_hat

    def get_embedding(self, x, edge_index):
        _, _, embeddings = self.forward(x, edge_index)
        return embeddings

class EdgeDyFraudNet(nn.Module):
    def __init__(self, input_dim, edge_attr_dim, memory_size=16, hidden_size=16, enable_memory=True, gnn_type="GCN",
                 num_layers=2,
                 dropout=0.0, heads=1):
        super().__init__()
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, hidden_size)
        if enable_memory:
            self.conv1 = EvolveGNN_O(hidden_size, memory_size, hidden_size, gnn_type)
            self.conv2 = EvolveGNN_O(hidden_size, memory_size, hidden_size, gnn_type)
        else:
            if gnn_type == "GAT":
                self.conv1 = pyg_nn.GATConv(hidden_size, hidden_size // heads, heads=heads)
                self.conv2 = pyg_nn.GATConv(hidden_size, hidden_size // heads, heads=heads)
            elif gnn_type == "GIN":
                self.conv1 = pyg_nn.GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
                self.conv2 = pyg_nn.GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
            else:  # GCN
                self.conv1 = pyg_nn.GCNConv(hidden_size, hidden_size)
                self.conv2 = pyg_nn.GCNConv(hidden_size, hidden_size)
        self.postprocessing1 = geom_nn.Linear(2 * hidden_size + edge_attr_dim, 1)
        self.postprocessing_anomaly = geom_nn.Linear(2 * hidden_size + edge_attr_dim, 1)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocessing1.reset_parameters()

    def forward(self, x, edge_index, edge_label_index, edge_attr):
        # Preprocess step
        # h_hat is embedding, h classification, h_anomaly is for anomaly score
        h = self.preprocess1(x)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.conv1(h, edge_index)

        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h, inplace=False)
        h_hat = F.dropout(h, p=self.dropout, inplace=True)
        h_src = h_hat[edge_label_index[0]]
        h_dst = h_hat[edge_label_index[1]]
        combined = torch.cat([h_src, h_dst, edge_attr], dim=1)
        h = self.postprocessing1(combined)
        # h = self.postprocessing1(h_hat)
        # CLS layer
        h = h.view(-1)
        # anom score
        anomaly_score = self.postprocessing_anomaly(combined).squeeze(-1)
        return h, anomaly_score, h_hat

    def get_embedding(self, x, edge_index):
        _, _, embeddings = self.forward(x, edge_index)
        return embeddings

class EvolveGNN_O(nn.Module):
    def __init__(self, in_channels, memory_size, out_channels, gnn_type="GIN", heads=1):
        super().__init__()

        self.gnn_type = gnn_type
        self.heads = heads  # Number of heads for GAT

        if gnn_type == "GAT":
            self.gnn = pyg_nn.GATConv(in_channels, out_channels // heads, heads=heads)
        elif gnn_type == "GIN":
            self.gnn = pyg_nn.GINConv(nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU()))
        else:
            self.gnn = pyg_nn.GCNConv(in_channels, out_channels)

        self.gru = nn.GRU(memory_size, memory_size)  # memory

        # memory to Weight Transforms
        if gnn_type == "GAT":
            self.weight_transform = nn.Linear(memory_size, in_channels * (out_channels // heads) * heads)
        elif gnn_type == "GIN":
            self.weight_transform = nn.Linear(memory_size, in_channels * out_channels)
        else:
            self.weight_transform = nn.Linear(memory_size, in_channels * out_channels)

        # memory initialization (with batch dimension), work as registry in GPU
        self.register_buffer("memory_weights", torch.zeros(1, memory_size))

    def forward(self, x, edge_index):
        # insuring both data and model are in same device, especially in GPU
        memory = self.memory_weights.to(x.device)
        update_memory, _ = self.gru(memory)
        update_memory = update_memory.squeeze(0)  # Remove batch dimension
        new_weights = self.weight_transform(update_memory)
        if self.gnn_type == "GAT":
            new_weights = new_weights.view(self.heads, self.gnn.in_channels, self.gnn.out_channels)
            for i in range(self.heads):
                self.gnn.att_l[i].weight = nn.Parameter(new_weights[i].detach())  # Update GAT attention head weights
        elif self.gnn_type == "GIN":
            new_weights = new_weights.view(self.gnn.nn[0].weight.shape)  # Update GIN MLP
            self.gnn.nn[0].weight = nn.Parameter(new_weights.detach())
        else:
            new_weights = new_weights.view(self.gnn.lin.weight.shape)
            self.gnn.lin.weight = nn.Parameter(new_weights.detach())

        self.memory_weights.copy_(update_memory.detach())

        out = self.gnn(x, edge_index)
        return out

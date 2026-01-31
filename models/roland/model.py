import torch
from dataclasses import dataclass
from torch.nn import GRUCell
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
import torch_geometric.nn as pyg_nn


class RolandGNN(torch.nn.Module):
    def __init__(self, input_dim, num_layers, hidden_size, num_nodes, previous_embeddings, dropout=0.0,
                 gnn_type="GCN", update='gru',
                 heads=1):
        # """
        # Args:
        #     input_dim: Dimension of input features
        #     hidden_conv1: Dimension of conv1
        #     hidden_conv2: Dimension of conv2
        #     c_out: Dimension of the output features. Usually number of classes in classification
        #     num_layers: Number of "hidden" graph layers
        #     layer_name: String of the graph layer to use
        #     dp_rate: Dropout rate to apply throughout the network
        #     role_embedding: for concatenation of role_embedding at first layer
        #     kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        # """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        # TODO: I should find a solution for handling multiple layer forward
        self.hidden_size = hidden_size
        self.preprocess1 = Linear(input_dim, 128)
        self.preprocess2 = Linear(128, hidden_size)
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            if gnn_type == "GAT":
                self.convs.append(pyg_nn.GATConv(hidden_size, hidden_size // heads, heads=heads))
            elif gnn_type == "GIN":
                self.convs.append(pyg_nn.GINConv(
                    nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())))
            else:  # GCN
                self.convs.append(pyg_nn.GCNConv(hidden_size, hidden_size))
        self.postprocessing1 = pyg_nn.Linear(hidden_size, 1)
        self.postprocessing_anomaly = pyg_nn.Linear(hidden_size, 1)
        self.dropout = dropout
        # Update layer
        self.update = update
        self.register_buffer("tau", torch.tensor([0.0]))
        if self.update == "moving":
            self.tau = torch.Tensor([0])
        elif self.update == "gru":
            self.gru_cells = nn.ModuleList(
                [GRUCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)])
        elif self.update == "mlp":
            self.mlp_layers = nn.ModuleList(
                [pyg_nn.Linear(self.hidden_size * 2, self.hidden_size) for _ in range(self.num_layers)])
        else:
            assert (0 <= self.update <= 1)
            self.tau = torch.Tensor([self.update])
        for i in range(num_layers):
            self.register_buffer(f"previous_embeddings{i}", previous_embeddings[i].clone().detach())

    def set_previous_embeddings(self, previous_embeddings):
        with torch.no_grad():
            for i, emb in enumerate(previous_embeddings):
                getattr(self, f"previous_embeddings{i}").copy_(emb.clone().detach())

    def set_tau(self, new_tau):
        if not hasattr(self, "tau"):
            raise AttributeError("Tau is not defined for this update type.")
        self.tau.fill_(new_tau)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.postprocessing1.reset_parameters()
        self.postprocessing_anomaly.reset_parameters()

    def forward(self, x, edge_index):
        current_embeddings = [torch.Tensor([]) for _ in range(self.num_layers)]

        # Preprocess step
        h = self.preprocess1(x)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)

        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.leaky_relu(h, inplace=False)  # I should check whether doing inplace here is safe or not
            h = F.dropout(h, p=self.dropout, inplace=True)
            # Update embedding after first layer
            if self.update == "gru":
                h = self.gru_cells[i](h, getattr(self, f"previous_embeddings{i}"))
            elif self.update == "mlp":
                hin = torch.cat((h, getattr(self, f"previous_embeddings{i}")), dim=1)
                h = self.mlp_layers[i](hin)
            else:
                prev = getattr(self, f"previous_embeddings{i}")
                h = self.tau * prev + (1 - self.tau) * h
            current_embeddings[i] = h.clone()
        out = self.postprocessing1(h)
        out = out.view(-1)
        anomaly_score = self.postprocessing_anomaly(h).squeeze(-1)
        return out, anomaly_score, current_embeddings, h

    def get_embedding(self, x, edge_index):
        _, _, _, embeddings = self.forward(x, edge_index)
        return embeddings


class EdgeRolandGNN(torch.nn.Module):
    def __init__(self, input_dim, num_layers, hidden_size, num_nodes, previous_embeddings, edge_attr_dim,
                 dropout=0.0, gnn_type="GCN",
                 update='gru', heads=1):

        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        # TODO: I should find a solution for handling multiple layer forward
        self.hidden_size = hidden_size
        self.preprocess1 = Linear(input_dim, 128)
        self.preprocess2 = Linear(128, hidden_size)
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            if gnn_type == "GAT":
                self.convs.append(pyg_nn.GATConv(hidden_size, hidden_size // heads, heads=heads))
            elif gnn_type == "GIN":
                self.convs.append(pyg_nn.GINConv(
                    nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())))
            else:  # GCN
                self.convs.append(pyg_nn.GCNConv(hidden_size, hidden_size))
        self.postprocessing1 = pyg_nn.Linear(2 * hidden_size + edge_attr_dim, 1)
        self.postprocessing_anomaly = pyg_nn.Linear(2 * hidden_size + edge_attr_dim, 1)
        self.dropout = dropout
        # Update layer
        self.update = update
        self.register_buffer("tau", torch.tensor([0.0]))
        if self.update == "moving":
            self.tau = torch.Tensor([0])
        elif self.update == "gru":
            self.gru_cells = nn.ModuleList(
                [GRUCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)])
        elif self.update == "mlp":
            self.mlp_layers = nn.ModuleList(
                [pyg_nn.Linear(self.hidden_size * 2, self.hidden_size) for _ in range(self.num_layers)])
        else:
            assert (0 <= self.update <= 1)
            self.tau = torch.Tensor([self.update])
        for i in range(num_layers):
            self.register_buffer(f"previous_embeddings{i}", previous_embeddings[i].clone().detach())

    def set_previous_embeddings(self, previous_embeddings):
        with torch.no_grad():
            for i, emb in enumerate(previous_embeddings):
                # print(i,emb)
                getattr(self, f"previous_embeddings{i}").copy_(emb.clone().detach())

    def set_tau(self, new_tau):
        if not hasattr(self, "tau"):
            raise AttributeError("Tau is not defined for this update type.")
        self.tau.fill_(new_tau)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.postprocessing1.reset_parameters()

    def forward(self, x, edge_index, edge_label_index, edge_attr, num_current_edges=None,
                num_previous_edges=None):
        current_embeddings = [torch.Tensor([]) for _ in range(self.num_layers)]
        # Preprocess step
        h = self.preprocess1(x)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)

        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.leaky_relu(h, inplace=False)  # I should check whether doing inplace here is safe or not
            h = F.dropout(h, p=self.dropout, inplace=True)
            # Update embedding after first layer
            if self.update == "gru":
                h = self.gru_cells[i](h, getattr(self, f"previous_embeddings{i}"))
            elif self.update == "mlp":
                hin = torch.cat((h, getattr(self, f"previous_embeddings{i}")), dim=1)
                h = self.mlp_layers[i](hin)
            else:
                prev = getattr(self, f"previous_embeddings{i}")
                h = self.tau * prev + (1 - self.tau) * h
            current_embeddings[i] = h.clone()

        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        # HADAMARD
        # h_hadamard = torch.mul(h_src, h_dst)
        # edge_attr = edge_features[edge_label_index]
        combined = torch.cat([h_src, h_dst, edge_attr], dim=1)
        out = self.postprocessing1(combined)
        out = out.view(-1)
        anomaly_score = self.postprocessing_anomaly(combined).squeeze(-1)
        return out, anomaly_score, current_embeddings

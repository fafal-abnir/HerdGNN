import torch
from dataclasses import dataclass
from torch.nn import GRUCell
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
import torch_geometric.nn as pyg_nn


@dataclass
class ModelConfig:
    input_dim: int
    hidden_conv1: int
    hidden_conv2: int
    num_nodes: int
    dropout: float = 0
    gnn_name: str = "GCN"
    update: str = "moving"


class RolandGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_conv1, hidden_conv2, num_nodes, dropout=0.0, gnn_type="GCN", update='gru',
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
        # TODO: I should find a solution for handling multiple layer forward
        self.hidden_conv1 = hidden_conv1
        self.hidden_conv2 = hidden_conv2
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, 128)
        if gnn_type == "GAT":
            self.conv1 = pyg_nn.GATConv(128, hidden_conv1 // heads, heads=heads)
            self.conv2 = pyg_nn.GATConv(hidden_conv1, hidden_conv2 // heads, heads=heads)
        elif gnn_type == "GIN":
            self.conv1 = pyg_nn.GINConv(nn.Sequential(nn.Linear(128, hidden_conv1), nn.ReLU()))
            self.conv2 = pyg_nn.GINConv(nn.Sequential(nn.Linear(hidden_conv1, hidden_conv2), nn.ReLU()))
        else:  # GCN
            self.conv1 = pyg_nn.GCNConv(128, hidden_conv1)
            self.conv2 = pyg_nn.GCNConv(hidden_conv1, hidden_conv2)
        # self.conv1 = gnn_layer(128, hidden_conv1)
        # self.conv2 = gnn_layer(hidden_conv1, hidden_conv2)
        self.postprocessing1 = pyg_nn.Linear(hidden_conv2, 2)
        self.dropout = dropout
        # Update layer
        self.update = update
        if self.update == "moving":
            self.tau = torch.Tensor([0])
        elif self.update == "gru":
            self.gru1 = GRUCell(self.hidden_conv1, self.hidden_conv1)
            self.gru2 = GRUCell(self.hidden_conv2, self.hidden_conv2)
        elif self.update == "mlp":
            self.mlp1 = pyg_nn.Linear(self.hidden_conv1 * 2, self.hidden_conv1)
            self.mlp2 = pyg_nn.Linear(self.hidden_conv2 * 2, self.hidden_conv2)
        else:
            assert (0 <= self.update <= 1)
            self.tau = torch.Tensor([self.update])
        # find better soluction for this part
        self.previous_embeddings = [
            torch.Tensor([[0 for _ in range(self.hidden_conv1)] for _ in range(self.num_nodes)]),
            torch.Tensor([[0 for _ in range(self.hidden_conv2)] for _ in range(self.num_nodes)])]

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocessing1.reset_parameters()

    def forward(self, x, edge_index, previous_embeddings=None):
        if previous_embeddings is not None:
            self.previous_embeddings = [previous_embeddings[0].clone(), previous_embeddings[1].clone()]
        current_embeddings = [torch.Tensor([]), torch.Tensor([])]
        # Preprocess step
        h = self.preprocess1(x)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)

        h = self.conv1(h, edge_index)
        h = F.leaky_relu(h, inplace=False)  # I should check whether doing inplace here is safe or not
        h = F.dropout(h, p=self.dropout, inplace=True)
        # Update embedding after first layer
        if self.update == "gru":
            h = self.gru1(h, self.previous_embeddings[0].clone())
        elif self.update == "mlp":
            hin = torch.cat((h, self.previous_embeddings[0].clone()), dim=1)
            h = self.mlp1(hin)
        else:
            h = torch.Tensor(
                (self.tau * self.previous_embeddings[0].clone() + (1 - self.tau) * h.clone()).detach().numpy())
        current_embeddings[0] = h.clone()
        # Conv2 layer 2
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        if self.update == "gru":
            h = self.gru2(h, self.previous_embeddings[1].clone())
        elif self.update == "mlp":
            hin = torch.cat((h, self.previous_embeddings[1].clone()), dim=1)
            h = self.mlp2(hin)
        else:
            h = torch.Tensor((self.tau * self.previous_embeddings[1].clone() + (1 - self.tau) * h.clone()))
        current_embeddings[1] = h.clone()

        # HADAMARD
        # h_src = h[edge_label_index[0]]
        # h_dst = h[edge_label_index[1]]
        # h_hadamard = torch.mul(h_src, h_dst)
        h = self.postprocessing1(h)
        h = torch.sum(h, dim=-1)
        return h, current_embeddings


class EdgeRolandGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_conv1, hidden_conv2, num_nodes, edge_attr_dim, dropout=0.0, gnn_type="GCN",
                 update='gru',heads=1):
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
        # TODO: I should find a solution for handling multiple layer forward
        self.hidden_conv1 = hidden_conv1
        self.hidden_conv2 = hidden_conv2
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, 128)
        if gnn_type == "GAT":
            self.conv1 = pyg_nn.GATConv(128, hidden_conv1 // heads, heads=heads)
            self.conv2 = pyg_nn.GATConv(hidden_conv1, hidden_conv2 // heads, heads=heads)
        elif gnn_type == "GIN":
            self.conv1 = pyg_nn.GINConv(nn.Sequential(nn.Linear(128, hidden_conv1), nn.ReLU()))
            self.conv2 = pyg_nn.GINConv(nn.Sequential(nn.Linear(hidden_conv1, hidden_conv2), nn.ReLU()))
        else:  # GCN
            self.conv1 = pyg_nn.GCNConv(128, hidden_conv1)
            self.conv2 = pyg_nn.GCNConv(hidden_conv1, hidden_conv2)
        self.postprocessing1 = Linear(2 * hidden_conv2 + edge_attr_dim, 1)
        self.dropout = dropout
        self.update = update
        # Update layer
        self.update = self.update
        if self.update == "moving":
            self.tau = torch.Tensor([0])
        elif self.update == "gru":
            self.gru1 = GRUCell(self.hidden_conv1, self.hidden_conv1)
            self.gru2 = GRUCell(self.hidden_conv2, self.hidden_conv2)
        elif self.update == "mlp":
            self.mlp1 = pyg_nn.Linear(self.hidden_conv1 * 2, self.hidden_conv1)
            self.mlp2 = pyg_nn.Linear(self.hidden_conv2 * 2, self.hidden_conv2)
        else:
            assert (0 <= self.update <= 1)
            self.tau = torch.Tensor([self.update])
        # find better soluction for this part
        self.previous_embeddings = [
            torch.Tensor([[0 for _ in range(self.hidden_conv1)] for _ in range(self.num_nodes)]),
            torch.Tensor([[0 for _ in range(self.hidden_conv2)] for _ in range(self.num_nodes)])]

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocessing1.reset_parameters()

    def forward(self, x, edge_index, edge_label_index, edge_attr, previous_embeddings=None, num_current_edges=None,
                num_previous_edges=None):
        if previous_embeddings is not None:
            self.previous_embeddings = [previous_embeddings[0].clone(), previous_embeddings[1].clone()]
        if self.update == "moving" and num_current_edges is not None and num_previous_edges is not None:
            self.tau = torch.Tensor([num_previous_edges / (num_previous_edges + num_current_edges)]).clone()
        current_embeddings = [torch.Tensor([]), torch.Tensor([])]
        # Preprocess step
        h = self.preprocess1(x)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        h = self.preprocess2(h)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)

        h = self.conv1(h, edge_index)
        h = F.leaky_relu(h, inplace=False)  # I should check whether doing inplace here is safe or not
        h = F.dropout(h, p=self.dropout, inplace=True)
        # Update embedding after first layer
        if self.update == "gru":
            h = self.gru1(h, self.previous_embeddings[0].clone())
        elif self.update == "mlp":
            hin = torch.cat((h, self.previous_embeddings[0].clone()), dim=1)
            h = self.mlp1(hin)
        else:
            h = torch.Tensor(
                (self.tau * self.previous_embeddings[0].clone() + (1 - self.tau) * h.clone()).detach().numpy())
        current_embeddings[0] = h.clone()
        # Conv2 layer 2
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h, inplace=False)
        h = F.dropout(h, p=self.dropout, inplace=True)
        if self.update == "gru":
            h = self.gru2(h, self.previous_embeddings[1].clone())
        elif self.update == "mlp":
            hin = torch.cat((h, self.previous_embeddings[1].clone()), dim=1)
            h = self.mlp2(hin)
        else:
            h = torch.Tensor((self.tau * self.previous_embeddings[1].clone() + (1 - self.tau) * h.clone()))
        current_embeddings[1] = h.clone()

        # HADAMARD
        h_src = h[edge_label_index[0]]
        h_dst = h[edge_label_index[1]]
        # h_hadamard = torch.mul(h_src, h_dst)
        # edge_attr = edge_features[edge_label_index]
        combined = torch.cat([h_src, h_dst, edge_attr], dim=1)
        h = self.postprocessing1(combined)
        h = h.view(-1)
        return h, current_embeddings

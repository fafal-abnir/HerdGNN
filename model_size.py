import argparse
import torch
from termcolor import colored
from colorama import init
import copy
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from datasets.data_loading import get_dataset
from torch_geometric.data import DataLoader
from models.dyfraudnet.model import NodeDyFraudNet, EdgeDyFraudNet
from models.roland.model import RolandGNN, EdgeRolandGNN
from models.dyfraudnet.lightning_modules import LightningNodeGNN, LightningEdgeGNN
from datetime import datetime
from utils.visualization import visualize_embeddings

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
init()

def get_args():
    parser = argparse.ArgumentParser(description="DyFraudNetGNN Training Arguments")
    parser.add_argument("--enable_memory", action="store_true", help="Enable the memory for GNN")
    parser.add_argument("--fresh_start", action="store_true", help="retraining from scratch on each timestamp")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate(default:0.01")
    parser.add_argument("--alpha", type=float, default=0.1, help="weight of deviation loss to addup to loss function")
    parser.add_argument("--anomaly_loss_margin", type=float, default=4.0, help="Anomaly loss margin")
    parser.add_argument("--blend_factor", type=float, default=0.9, help="blend factor for merging 2 distribution")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of hidden layers (default: 128)")
    parser.add_argument("--memory_size", type=int, default=128,
                        help="Size of memory for evolving weights (default: 128)")
    parser.add_argument("--gnn_type", type=str, choices=["GIN", "GAT", "GCN"], default="GCN",
                        help="Type of GNN model: GIN, GAT, or GCN (default: GCN)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--dataset_name", type=str,
                        choices=["EllipticPP", "DGraphFin", "BitcoinOTC", "MOOC", "RedditTitle",
                                 "RedditBody", "EthereumPhishing", "SAMLSim"], default="RedditTitle")
    parser.add_argument("--force_reload_dataset", action="store_true", help="Force to download the dataset again.")
    parser.add_argument("--graph_window_size", type=str, choices=["day", "week", "month"], default="month",
                        help="the size of graph window size")
    parser.add_argument("--num_windows", type=int, default=10, help="Number of windows for running the experiment")
    parser.add_argument("--embedding_visualization", action="store_true",
                        help="Visualization of train data before and after training")
    return parser.parse_args()


def count_model_elements(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "buffers": buffers,
        "all_state_dict": total_params + buffers,
    }


def print_model_params(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    print(colored(count_model_elements(model), "green"))
    total_size = (param_size + buffer_size) / (1024 ** 2)
    print(f"Model size (parameters + buffers): {total_size:.2f} MB")

def main():
    args = get_args()
    # Model arguments
    enable_memory = args.enable_memory
    fresh_start = args.fresh_start
    embedding_visualization = args.embedding_visualization
    hidden_size = args.hidden_size
    memory_size = args.memory_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    alpha = args.alpha
    anomaly_loss_margin = args.anomaly_loss_margin
    blend_factor = args.blend_factor
    gnn_type = args.gnn_type
    num_layers = args.num_layers
    dropout = args.dropout
    # Data arguments
    dataset_name = args.dataset_name
    force_reload_dataset = args.force_reload_dataset
    graph_window_size = args.graph_window_size
    num_windows = args.num_windows
    model = None
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    print(colored(f"Code params:{vars(args)}","yellow"))
    if dataset_name in ["DGraphFin", "EllipticPP", "EthereumPhishing"]:
        task = "Node"
        lightning_root_dir = "experiments/dyfraudnet/node_level"
        if dataset_name == "EllipticPP":
            graph_window_size = "hour"
    else:
        task = "Edge"
        lightning_root_dir = "experiments/dyfraudnet/edge_level"
    dataset = get_dataset(name=dataset_name, force_reload=force_reload_dataset, edge_window_size=graph_window_size,
                          num_windows=num_windows)
    print(colored(f"Number of windows: {len(dataset)}","blue"))
    num_nodes = dataset.num_nodes
    previous_embeddings = [torch.zeros((num_nodes, hidden_size)) for _ in range(num_layers)]
    if task == "Node":
        print("Our method with memory False:")
        model = NodeDyFraudNet(dataset[0].x.shape[1], memory_size=memory_size, hidden_size=hidden_size,
                               out_put_size=2, dropout=dropout, num_layers=num_layers,
                               gnn_type=gnn_type, enable_memory=False)
        print_model_params(model)
        print("Our method with memory True:")
        model = NodeDyFraudNet(dataset[0].x.shape[1], memory_size=memory_size, hidden_size=hidden_size,
                               out_put_size=2, dropout=dropout, num_layers=num_layers,
                               gnn_type=gnn_type, enable_memory=True)
        print_model_params(model)
        print("Roland method with gru")
        model = RolandGNN(dataset[0].x.shape[1], num_layers, hidden_size, dataset.num_nodes,
                          previous_embeddings,
                          gnn_type=gnn_type,
                          dropout=dropout,
                          update='gru')
        print_model_params(model)
        print("Roland method with mlp")
        model = RolandGNN(dataset[0].x.shape[1], num_layers, hidden_size, dataset.num_nodes,
                          previous_embeddings,
                          gnn_type=gnn_type,
                          dropout=dropout,
                          update='mlp')
        print_model_params(model)
        print("Roland method with moving")
        model = RolandGNN(dataset[0].x.shape[1], num_layers, hidden_size, dataset.num_nodes,
                          previous_embeddings,
                          gnn_type=gnn_type,
                          dropout=dropout,
                          update='moving')
        print_model_params(model)
    else:
        print("Our method with memory False:")
        model = EdgeDyFraudNet(dataset[0].x.shape[1], edge_attr_dim=dataset.num_edge_features,
                               num_layers=num_layers, dropout=dropout,
                               memory_size=memory_size, hidden_size=hidden_size,
                               gnn_type=gnn_type, enable_memory=False)
        print_model_params(model)
        print("Our method with memory True:")
        model = EdgeDyFraudNet(dataset[0].x.shape[1], edge_attr_dim=dataset.num_edge_features,
                               num_layers=num_layers, dropout=dropout,
                               memory_size=memory_size, hidden_size=hidden_size,
                               gnn_type=gnn_type, enable_memory=True)
        print_model_params(model)
        print("Roland method with gru")
        model = EdgeRolandGNN(dataset[0].x.shape[1], num_layers, hidden_size,
                              dataset.num_nodes, previous_embeddings,
                              dataset.num_edge_features, gnn_type=gnn_type,
                              dropout=dropout,
                              update="gru")
        print_model_params(model)
        print("Roland method with mlp")
        model = EdgeRolandGNN(dataset[0].x.shape[1], num_layers, hidden_size,
                              dataset.num_nodes, previous_embeddings,
                              dataset.num_edge_features, gnn_type=gnn_type,
                              dropout=dropout,
                              update="mlp")
        print_model_params(model)
        print("Roland method with moving")
        model = EdgeRolandGNN(dataset[0].x.shape[1], num_layers, hidden_size,
                              dataset.num_nodes, previous_embeddings,
                              dataset.num_edge_features, gnn_type=gnn_type,
                              dropout=dropout,
                              update="moving")
        print_model_params(model)
if __name__ == "__main__":
    main()

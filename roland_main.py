import argparse
import torch
import copy
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from datasets.data_loading import get_dataset
from torch_geometric.data import DataLoader
from models.roland.model import RolandGNN, EdgeRolandGNN
from models.roland.lightning_modules import LightningNodeGNN, LightningEdgeGNN
from datetime import datetime

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser(description="Roland Training Arguments")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate(default:0.001")
    parser.add_argument("--hidden_conv1", type=int, default=16, help="Size of hidden layers (default: 128)")
    parser.add_argument("--hidden_conv2", type=int, default=16,
                        help="Size of memory for evolving weights (default: 128)")
    parser.add_argument("--gnn_type", type=str, choices=["GIN", "GAT", "GCN"], default="GCN",
                        help="Type of GNN model: GIN, GAT, or GCN (default: GCN)")
    parser.add_argument("--update_type", type=str, choices=["gru", "mlp", "moving"], default="gru",
                        help="Type of updating node embeddings: gru, mlp, or moving (default: gru)")
    parser.add_argument("--dataset_name", type=str,
                        choices=["DGraphFin", "BitcoinOTC", "MOOC", "RedditTitle", "RedditBody"], default="RedditTitle")
    parser.add_argument("--force_reload_dataset", action="store_true", help="Force to download the dataset again.")
    parser.add_argument("--graph_window_size", type=str, choices=["day", "week", "month"], default="month",
                        help="the size of graph window size")
    parser.add_argument("--num_windows", type=int, default=10, help="Number of windows for running the experiment")
    return parser.parse_args()


def main():
    args = get_args()
    hidden_conv1 = args.hidden_conv1
    hidden_conv2 = args.hidden_conv2
    epochs = args.epochs
    learning_rate = args.learning_rate
    gnn_type = args.gnn_type
    update_type = args.update_type
    dataset_name = args.dataset_name
    graph_window_size = args.graph_window_size
    num_windows = args.num_windows
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if dataset_name in ["DGraphFin"]:
        task = "Node"
        lightning_root_dir = "experiments/roland/node_level"
    else:
        task = "Edge"
        lightning_root_dir = "experiments/roland/edge_level"
    dataset = get_dataset(name=dataset_name, force_reload=False, edge_window_size=graph_window_size,
                          num_windows=num_windows)
    # dataset = DGraphFin('data/DGraphFin', force_reload=True, edge_window_size=graph_window_size,
    #                     num_windows=num_windows)
    for data_index in range(len(dataset) - 1):
        if data_index == 0:
            num_nodes = dataset.num_nodes
            previous_embeddings = [
                torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(num_nodes)]),
                torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(num_nodes)])]
        else:
            _, previous_embeddings = lightningModule.forward(train_data)
        snapshot = dataset[data_index]
        if snapshot.x is None:
            snapshot.x = torch.Tensor([[1] for _ in range(snapshot.num_nodes)])

        if snapshot.x is None:
            snapshot.x = torch.Tensor([[1] for _ in range(snapshot.num_nodes)])

        if task == "Node":
            train_mask = torch.zeros_like(snapshot.node_mask, dtype=torch.bool)
            val_mask = torch.zeros_like(snapshot.node_mask, dtype=torch.bool)
            train_indices = snapshot.node_mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(train_indices))
            split_idx = int(0.9 * len(train_indices))
            train_mask[train_indices[perm[:split_idx]]] = True
            val_mask[train_indices[perm[split_idx:]]] = True
            train_data = snapshot.clone()
            train_data.node_mask = train_mask
            train_data.previous_embeddings = previous_embeddings

            val_data = snapshot.clone()
            val_data.node_mask = val_mask
            test_data = copy.deepcopy(dataset[data_index + 1])
            test_data.num_current_edges = test_data.num_edges
            test_data.num = test_data.num_nodes
            val_data.previous_embeddings = previous_embeddings
            test_data.previous_embeddings = previous_embeddings
            if snapshot.x is None:
                test_data.x = torch.Tensor([[1] for _ in range(test_data.num_nodes)])

            model = RolandGNN(snapshot.x.shape[1], hidden_conv1, hidden_conv2, dataset.num_nodes, gnn_type=gnn_type,
                              update=update_type)
            lightningModule = LightningNodeGNN(model, learning_rate=learning_rate)
            experiments_dir = f"{lightning_root_dir}/{dataset_name}/{graph_window_size}/{gnn_type}_{update_type}_{hidden_conv1}_{hidden_conv2}/{experiment_datetime}/index_{data_index} "
            csv_logger = CSVLogger(experiments_dir, version="")
            print(f"Time Index: {data_index}, data: {dataset_name}")
            print(train_data)
            print(val_data)
            print(test_data)
            # Start training and testing.
            train_loader = DataLoader([train_data], batch_size=1)
            val_loader = DataLoader([val_data], batch_size=1)
            test_loader = DataLoader([test_data], batch_size=1)
            trainer = L.Trainer(default_root_dir=experiments_dir,
                                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_avg_pr")],
                                accelerator="auto",
                                devices="auto",
                                enable_progress_bar=True,
                                logger=csv_logger,
                                max_epochs=epochs
                                )
            trainer.fit(lightningModule, train_loader, val_loader)
            trainer.test(lightningModule, test_loader)
        else:
            num_edges = snapshot.edge_index.size(1)
            perm = torch.randperm(num_edges)
            train_ratio = 0.8
            val_ratio = 0.2
            train_end = int(train_ratio * num_edges)
            val_end = int((train_ratio + val_ratio) * num_edges)

            train_idx = perm[:train_end]
            val_idx = perm[train_end:val_end]
            # test_idx = perm[val_end:]

            train_data = snapshot.clone()
            train_data.edge_label_index = snapshot.edge_index[:, train_idx]
            train_data.edge_attr = snapshot.edge_attr[train_idx]
            train_data.y = train_data.y[train_idx]
            train_data.previous_embeddings = previous_embeddings

            val_data = snapshot.clone()
            val_data.edge_label_index = snapshot.edge_index[:, val_idx]
            val_data.y = snapshot.y[val_idx]
            val_data.edge_attr = snapshot.edge_attr[val_idx]

            test_data = copy.deepcopy(dataset[data_index + 1])
            test_data.num_current_edges = test_data.num_edges
            test_data.num = test_data.num_nodes
            val_data.previous_embeddings = previous_embeddings
            test_data.edge_label_index = test_data.edge_index
            test_data.previous_embeddings = previous_embeddings
            if snapshot.x is None:
                test_data.x = torch.Tensor([[1] for _ in range(test_data.num_nodes)])
            model = EdgeRolandGNN(snapshot.x.shape[1], hidden_conv1, hidden_conv2, dataset.num_nodes,
                                  dataset.num_edge_features, gnn_type=gnn_type,
                                  update=update_type)
            lightningModule = LightningEdgeGNN(model, learning_rate=learning_rate)
            experiments_dir = f"{lightning_root_dir}/{dataset_name}/{graph_window_size}/{gnn_type}_{update_type}_{hidden_conv1}_{hidden_conv2}/{experiment_datetime}/index_{data_index} "
            csv_logger = CSVLogger(experiments_dir, version="")
            print(f"Time Index: {data_index}, data: {dataset_name}")
            print(train_data)
            print(val_data)
            print(test_data)
            # Start training and testing.
            train_loader = DataLoader([train_data], batch_size=1)
            val_loader = DataLoader([val_data], batch_size=1)
            test_loader = DataLoader([test_data], batch_size=1)
            trainer = L.Trainer(default_root_dir=experiments_dir,
                                callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_avg_pr")],
                                accelerator="auto",
                                devices="auto",
                                enable_progress_bar=True,
                                logger=csv_logger,
                                max_epochs=epochs
                                )
            trainer.fit(lightningModule, train_loader, val_loader)
            trainer.test(lightningModule, test_loader)


if __name__ == "__main__":
    main()

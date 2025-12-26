import gc
import argparse
import torch
import copy
from termcolor import colored
from colorama import init
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import NeighborLoader

from datasets.data_loading import get_dataset
from torch_geometric.data import DataLoader
from models.roland.model import RolandGNN, EdgeRolandGNN
from models.roland.lightning_modules import LightningNodeGNN, LightningEdgeGNN
from datetime import datetime

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
init()


def get_args():
    parser = argparse.ArgumentParser(description="Roland Training Arguments")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate(default:0.001")
    parser.add_argument("--alpha", type=float, default=0.1, help="weight of deviation loss to addup to loss function")
    parser.add_argument("--anomaly_loss_margin", type=float, default=4.0, help="Anomaly loss margin")
    parser.add_argument("--blend_factor", type=float, default=0.9, help="blend factor for merging 2 distribution")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in roland")
    parser.add_argument("--hidden_size", type=int, default=16, help="Size of hidden layers (default: 16)")
    parser.add_argument("--gnn_type", type=str, choices=["GIN", "GAT", "GCN"], default="GCN",
                        help="Type of GNN model: GIN, GAT, or GCN (default: GCN)")
    parser.add_argument("--fresh_start", action="store_true", help="retraining from scratch on each timestamp")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--update_type", type=str, choices=["gru", "mlp", "moving"], default="gru",
                        help="Type of updating node embeddings: gru, mlp, or moving (default: gru)")
    parser.add_argument("--dataset_name", type=str,
                        choices=["EllipticPP", "DGraphFin", "BitcoinOTC", "MOOC",
                                 "RedditTitle", "RedditBody", "EthereumPhishing", "SAMLSim",
                                 "AMLWorldLarge", "AMLWorldMedium", "AMLWorldSmall"], default="RedditTitle")
    parser.add_argument("--force_reload_dataset", action="store_true", help="Force to download the dataset again.")
    parser.add_argument("--graph_window_size", type=str, choices=["day", "week", "month"], default="month",
                        help="the size of graph window size")
    parser.add_argument("--num_windows", type=int, default=10, help="Number of windows for running the experiment")
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


def main():
    args = get_args()
    # Model arguments
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    alpha = args.alpha
    anomaly_loss_margin = args.anomaly_loss_margin
    blend_factor = args.blend_factor
    gnn_type = args.gnn_type
    update_type = args.update_type
    fresh_start = args.fresh_start
    dropout = args.dropout
    # Data arguments
    dataset_name = args.dataset_name
    force_reload_dataset = args.force_reload_dataset
    graph_window_size = args.graph_window_size
    num_windows = args.num_windows
    model = None
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if dataset_name in ["DGraphFin", "EllipticPP"]:
        task = "Node"
        lightning_root_dir = "experiments/roland/node_level"
        if dataset_name == "EllipticPP":
            graph_window_size = "hour"
    else:
        task = "Edge"
        lightning_root_dir = "experiments/roland/edge_level"
    dataset = get_dataset(name=dataset_name, force_reload=force_reload_dataset, edge_window_size=graph_window_size,
                          num_windows=num_windows)
    print(colored(f"Number of windows: {len(dataset)}", "blue"))
    for data_index in range(len(dataset) - 1):
        if data_index == 0:
            num_nodes = dataset.num_nodes
            previous_embeddings = [torch.zeros((num_nodes, hidden_size)) for _ in range(num_layers)]
        snapshot = dataset[data_index]

        if task == "Node":
            ## preparing train and val
            train_mask = torch.zeros_like(snapshot.node_mask, dtype=torch.bool)
            val_mask = torch.zeros_like(snapshot.node_mask, dtype=torch.bool)
            train_indices = snapshot.node_mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(train_indices))
            split_idx = int(0.9 * len(train_indices))
            train_mask[train_indices[perm[:split_idx]]] = True
            val_mask[train_indices[perm[split_idx:]]] = True
            train_data = snapshot.clone()
            train_data.node_mask = train_mask

            val_data = snapshot.clone()
            val_data.node_mask = val_mask
            if (model is None) or fresh_start:
                model = RolandGNN(snapshot.x.shape[1], num_layers, hidden_size, dataset.num_nodes,
                                  previous_embeddings,
                                  gnn_type=gnn_type,
                                  dropout=dropout,
                                  update=update_type)
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()

                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()

                total_size = (param_size + buffer_size) / (1024 ** 2)
                print(colored(count_model_elements(model), "green"))
                print(f"Model size (parameters + buffers): {total_size:.2f} MB")
            lightningModule = LightningNodeGNN(model, learning_rate=learning_rate, alpha=alpha,
                                               anomaly_loss_margin=anomaly_loss_margin, blend_factor=blend_factor)
            experiments_dir = f"{lightning_root_dir}/{dataset_name}/{graph_window_size}/{gnn_type}_{update_type}_{hidden_size}/{experiment_datetime}/index_{data_index}"
            csv_logger = CSVLogger(experiments_dir, version="")
            csv_logger.log_hyperparams(vars(args))
            print(colored(f"Time Index: {data_index}, data: {dataset_name}", "yellow"))
            print(train_data)
            print(val_data)
            # Start training
            train_loader = DataLoader([train_data], batch_size=1)
            val_loader = DataLoader([val_data], batch_size=1)
            # early_stop_callback = EarlyStopping(
            #     monitor='val_loss',
            #     mode='min',
            #     patience=10
            # )
            # model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_avg_pr")
            checkpoint_callback = ModelCheckpoint(
                monitor="val_avg_pr",
                mode="max",
                save_top_k=1,
                save_weights_only=True,
                dirpath=experiments_dir,
                filename="best-checkpoint"
            )
            trainer = L.Trainer(default_root_dir=experiments_dir,
                                accelerator="auto",
                                devices="auto",
                                enable_progress_bar=True,
                                logger=csv_logger,
                                max_epochs=epochs,
                                callbacks=[checkpoint_callback]
                                )
            trainer.fit(lightningModule, train_loader, val_loader)
            _, _, previous_embeddings = lightningModule.forward(train_data)
            model.set_previous_embeddings(previous_embeddings)
            # testing
            test_data = copy.deepcopy(dataset[data_index + 1])
            test_data.num_current_edges = test_data.num_edges
            test_data.num = test_data.num_nodes
            test_loader = DataLoader([test_data], batch_size=1)
            print(test_data)
            trainer.test(lightningModule, test_loader)
            model.set_tau((test_data.edge_index.size(1) + 1e-8) / (
                    train_data.edge_index.size(1) + test_data.edge_index.size(1) + 1e-8))
            del train_data, train_loader, val_data, val_loader, test_data, test_loader, snapshot, previous_embeddings
            gc.collect()
            torch.cuda.empty_cache()
        else:  # Edge task=="Edge"
            num_edges = snapshot.edge_index.size(1)
            perm = torch.randperm(num_edges)
            train_ratio = 0.8
            val_ratio = 0.2
            train_end = int(train_ratio * num_edges)
            val_end = int((train_ratio + val_ratio) * num_edges)

            train_idx = perm[:train_end]
            val_idx = perm[train_end:val_end]
            # prepare train,val
            train_data = snapshot.clone()
            train_data.edge_label_index = snapshot.edge_index[:, train_idx]
            train_data.edge_attr = snapshot.edge_attr[train_idx]
            train_data.y = train_data.y[train_idx]

            val_data = snapshot.clone()
            val_data.edge_label_index = snapshot.edge_index[:, val_idx]
            val_data.y = snapshot.y[val_idx]
            val_data.edge_attr = snapshot.edge_attr[val_idx]
            if (model is None) or fresh_start:
                model = EdgeRolandGNN(snapshot.x.shape[1], num_layers, hidden_size,
                                      dataset.num_nodes, previous_embeddings,
                                      dataset.num_edge_features, gnn_type=gnn_type,
                                      dropout=dropout,
                                      update=update_type)
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()

                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()

                print(colored(count_model_elements(model), "green"))
                total_size = (param_size + buffer_size) / (1024 ** 2)
                print(f"Model size (parameters + buffers): {total_size:.2f} MB")
            lightningModule = LightningEdgeGNN(model, learning_rate=learning_rate, alpha=alpha,
                                               anomaly_loss_margin=anomaly_loss_margin, blend_factor=blend_factor)
            experiments_dir = f"{lightning_root_dir}/{dataset_name}/{graph_window_size}/{gnn_type}_{update_type}_{hidden_size}/{experiment_datetime}/index_{data_index}"
            csv_logger = CSVLogger(experiments_dir, version="")
            csv_logger.log_hyperparams(vars(args))
            print(f"Time Index: {data_index}, data: {dataset_name}")
            print(train_data)
            print(val_data)
            # Start training and testing.
            train_loader = DataLoader([train_data], batch_size=1)
            val_loader = DataLoader([val_data], batch_size=1)
            # early_stop_callback = EarlyStopping(
            #     monitor='val_loss',
            #     mode='min',
            #     patience=10
            # )
            # model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_avg_pr")
            checkpoint_callback = ModelCheckpoint(
                monitor="val_avg_pr",
                mode="max",
                save_top_k=1,
                save_weights_only=True,
                dirpath=experiments_dir,
                filename="best-checkpoint"
            )
            trainer = L.Trainer(default_root_dir=experiments_dir,
                                accelerator="auto",
                                devices="auto",
                                enable_progress_bar=True,
                                logger=csv_logger,
                                max_epochs=epochs,
                                callbacks=[checkpoint_callback]
                                )
            trainer.fit(lightningModule, train_loader, val_loader)
            _, _, previous_embeddings = lightningModule.forward(train_data)
            model.set_previous_embeddings(previous_embeddings)
            # Preparing test data
            test_data = copy.deepcopy(dataset[data_index + 1])
            test_data.num_current_edges = test_data.num_edges
            test_data.num = test_data.num_nodes
            test_data.edge_label_index = test_data.edge_index
            print(test_data)
            test_loader = DataLoader([test_data], batch_size=1)
            trainer.test(lightningModule, test_loader)
            model.set_tau((test_data.edge_index.size(1) + 1e-8) / (
                    train_data.edge_index.size(1) + test_data.edge_index.size(1) + 1e-8))
            del train_data, val_data, test_data, snapshot, previous_embeddings
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

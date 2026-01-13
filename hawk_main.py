import argparse
import torch
from termcolor import colored
from colorama import init
import copy
import torch
from torch_geometric.data import Data
from typing import List, Literal, Tuple

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from datasets.data_loading import get_dataset
from torch_geometric.data import DataLoader
from models.hawkes.model import NodeHawkGNN, EdgeHawkGNN
from models.hawkes.lightning_modules import LightningNodeGNN, LightningEdgeGNN
from datetime import datetime
from utils.visualization import visualize_embeddings

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
init()


def get_args():
    parser = argparse.ArgumentParser(description="DyFraudNetGNN Training Arguments")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 10)")
    parser.add_argument("--alpha", type=float, default=0.0, help="weight of deviation loss to addup to loss function")
    parser.add_argument("--anomaly_loss_margin", type=float, default=4.0, help="Anomaly loss margin")
    parser.add_argument("--blend_factor", type=float, default=1.0, help="blend factor for merging 2 distribution")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate(default:0.01")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of hidden layers (default: 128)")
    parser.add_argument("--gnn_type", type=str, choices=["GAT", "GCN"], default="GAT",
                        help="Type of HawkGNN model:  GAT or GCN (default: GCN)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--use_bn", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--dataset_name", type=str,
                        choices=["EllipticPP", "DGraphFin", "BitcoinOTC", "MOOC",
                                 "RedditTitle", "RedditBody", "EthereumPhishing", "SAMLSim",
                                 "AMLWorldLarge", "AMLWorldMedium", "AMLWorldSmall"], default="RedditTitle")
    parser.add_argument("--force_reload_dataset", action="store_true", help="Force to download the dataset again.")
    parser.add_argument("--hawk_window_size", type=int, default=6,
                        help="Length of window for hawkGNN training(T is for test, T-1 for val, the rest for train)")
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


def fuse_range(datasets: List[Data], start: int, end: int, class_lvl: Literal["Node", "Edge"] = "Node") -> Tuple[
    Data, Data]:
    """
    Fuse snapshots [start..end] into one PyG Data.
    - edge_age = (end - t)
    - x is taken from datasets[0].x (assumed same across all snapshots)
    - y is taken from datasets[0].y (if available)
    """
    assert len(datasets) >= 2, "Need at least 2 snapshots to make train/test."

    fused_edges, fused_attrs, fused_ages = [], [], []
    # Preparing train_data
    for t in range(start, end):
        data = datasets[t]
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # edge attributes (if missing, create empty tensor)
        edge_attr = data.edge_attr if data.edge_attr is not None else torch.zeros((num_edges, 0))
        edge_age = torch.full((num_edges, 1), fill_value=float(end - t))

        fused_edges.append(edge_index)
        fused_attrs.append(edge_attr)
        fused_ages.append(edge_age)

    flat_fused_attrs = []
    for x in fused_attrs:
        if isinstance(x, torch.Tensor):
            flat_fused_attrs.append(x)
        elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            flat_fused_attrs.append(x[0])
        else:
            raise TypeError(f"Unexpected type in fused_attrs: {type(x)}")

    # combine everything for train_data
    edge_index = torch.cat(fused_edges, dim=1)
    edge_attr = torch.cat(flat_fused_attrs)
    edge_ages = torch.cat(fused_ages)

    x = datasets[0].x
    train_y = None
    if class_lvl == "Node":
        node_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        for t in range(start, end):
            node_mask = torch.logical_or(node_mask, datasets[t].node_mask)
        train_y = datasets[0].y
        fused_train_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=train_y, edge_ages=edge_ages,
                                node_mask=node_mask)
    else:
        y_list = []
        for t in range(start, end):
            y_list.append(datasets[t].y)
        train_y = torch.cat(y_list, dim=0)
        fused_train_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=train_y, edge_ages=edge_ages)

    # Preparing test_data
    data = datasets[end]
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    edge_attr = data.edge_attr if data.edge_attr is not None else torch.zeros((num_edges, 0))
    edge_ages = torch.full((num_edges, 1), fill_value=float(0))

    x = datasets[end].x
    test_y = datasets[end].y
    if class_lvl == "Node":
        node_mask = datasets[end].node_mask
        fused_test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=test_y, edge_ages=edge_ages,
                               node_mask=node_mask)
    else:
        fused_test_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=test_y, edge_ages=edge_ages)
    return fused_train_data, fused_test_data


def main():
    args = get_args()
    # Model arguments
    embedding_visualization = args.embedding_visualization
    hidden_size = args.hidden_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    alpha = args.alpha
    anomaly_loss_margin = args.anomaly_loss_margin
    blend_factor = args.blend_factor
    gnn_type = args.gnn_type
    hawk_window_size = args.hawk_window_size
    num_layers = args.num_layers
    dropout = args.dropout
    use_bn = args.use_bn
    # Data arguments
    dataset_name = args.dataset_name
    force_reload_dataset = args.force_reload_dataset
    graph_window_size = args.graph_window_size
    num_windows = args.num_windows
    model = None
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if dataset_name in ["DGraphFin", "EllipticPP", "EthereumPhishing"]:
        task = "Node"
        lightning_root_dir = "experiments/hawkgnn/node_level"
        if dataset_name == "EllipticPP":
            graph_window_size = "hour"
    else:
        task = "Edge"
        lightning_root_dir = "experiments/hawkgnn/edge_level"
    dataset = get_dataset(name=dataset_name, force_reload=force_reload_dataset, edge_window_size=graph_window_size,
                          num_windows=num_windows)
    print(colored(f"Number of windows: {len(dataset)}", "blue"))
    for start_index in range(len(dataset) - hawk_window_size):
        experiments_dir = f"{lightning_root_dir}/{dataset_name}/{graph_window_size}/{gnn_type}_{hawk_window_size}/{experiment_datetime}/index_{start_index}"
        csv_logger = CSVLogger(experiments_dir, version="")
        if task == "Node":
            train_val_data, test_data = fuse_range(dataset, start_index,
                                                   start_index + hawk_window_size - 1)  # [t,t+win)
            train_mask = torch.zeros_like(train_val_data.node_mask, dtype=torch.bool)
            val_mask = torch.zeros_like(train_val_data.node_mask, dtype=torch.bool)
            train_indices = train_val_data.node_mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(train_indices))
            split_idx = int(0.8 * len(train_indices))
            train_mask[train_indices[perm[:split_idx]]] = True
            val_mask[train_indices[perm[split_idx:]]] = True
            train_data = train_val_data.clone()
            train_data.node_mask = train_mask

            val_data = train_val_data.clone()
            val_data.node_mask = val_mask
            # test_data.num_current_edges = test_data.num_edges
            # test_data.num = test_data.num_nodes
            if model is None:
                model = NodeHawkGNN(gnn_type=gnn_type, n_node=dataset.num_nodes,
                                    in_dim=train_val_data.x.shape[1], hid_dim=hidden_size,
                                    dropout=dropout, layers=num_layers,
                                    )
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()

                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()

                total_size = (param_size + buffer_size) / (1024 ** 2)
                print(colored(count_model_elements(model)), "green")
                print(f"Model size (parameters + buffers): {total_size:.2f} MB")
            else:
                model = NodeHawkGNN(gnn_type=gnn_type, n_node=dataset.num_nodes,
                                    in_dim=train_val_data.x.shape[1], hid_dim=hidden_size,
                                    dropout=dropout, layers=num_layers,
                                    )
            lightningModule = LightningNodeGNN(model, learning_rate=learning_rate, alpha=alpha,
                                               anomaly_loss_margin=anomaly_loss_margin, blend_factor=blend_factor)
            csv_logger.log_hyperparams(vars(args))
            print(colored(f"Time Index: {start_index}, hawk_window_size={hawk_window_size}, data: {dataset_name}",
                          "yellow"))
            print(train_data)
            print(val_data)
            print(test_data)
            # Start training and testing.
            train_loader = DataLoader([train_data], batch_size=1)
            val_loader = DataLoader([val_data], batch_size=1)
            test_loader = DataLoader([test_data], batch_size=1)
            # Callbacks
            # early_stop_callback = EarlyStopping(
            #     monitor='val_avg_pr',
            #     mode='max',
            #     patience=30
            # )
            checkpoint_callback = ModelCheckpoint(
                monitor="val_avg_pr",
                mode="max",
                save_top_k=1,
                save_weights_only=True,
                dirpath=experiments_dir,
                filename="best-checkpoint"
            )
            # model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_avg_pr")
            trainer = L.Trainer(default_root_dir=experiments_dir,
                                accelerator="auto",
                                devices="auto",
                                enable_progress_bar=True,
                                logger=csv_logger,
                                max_epochs=epochs,
                                callbacks=[checkpoint_callback]
                                )
            # Visualization embedding before training
            if embedding_visualization:
                print("start visualization")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                model.eval()
                train_data = train_data.to(device)
                train_labels = train_data.y[train_data.node_mask]
                train_labels_np = train_labels.cpu().numpy()
                with torch.no_grad():
                    all_embeddings = model.get_embedding(train_data.x, train_data.edge_index)
                    train_embeddings_np = all_embeddings[train_data.node_mask].cpu().numpy()
                train_label_colors = ['blue' if label == 0 else 'red' for label in train_labels_np]
                visualize_embeddings(train_embeddings_np, train_label_colors, 0,
                                     f'{experiments_dir}/train_embeddings_not_trained.png')

            trainer.fit(lightningModule, train_loader, val_loader)
            trainer.test(lightningModule, test_loader)

            # Visualization embedding
            if embedding_visualization:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                model.eval()
                train_data = train_data.to(device)
                test_data = test_data.to(device)
                test_labels = test_data.y[test_data.node_mask]
                test_labels_np = test_labels.cpu().numpy()
                with torch.no_grad():
                    train_embeddings = model.get_embedding(train_data.x, train_data.edge_index)
                    train_embeddings_np = train_embeddings[train_data.node_mask].cpu().numpy()
                    test_embeddings = model.get_embedding(test_data.x, test_data.edge_index)
                    test_embeddings_np = test_embeddings[test_data.node_mask].cpu().numpy()
                # train_label_colors = ['blue' if label == 0 else 'red' for label in train_data.y.cpu().numpy()]
                test_label_colors = ['blue' if label == 0 else 'red' for label in test_labels_np]
                visualize_embeddings(train_embeddings_np, train_label_colors, epochs,
                                     f'{experiments_dir}/train_embedding_trained.png')
                visualize_embeddings(test_embeddings_np, test_label_colors, 'None',
                                     f'{experiments_dir}/test_embedding.png')

        else:
            train_val_data, test_data = fuse_range(dataset, start_index, start_index + hawk_window_size - 1,
                                                   class_lvl="Edge")
            num_edges = train_val_data.edge_index.size(1)
            perm = torch.randperm(num_edges)
            train_ratio = 0.8
            val_ratio = 0.2
            train_end = int(train_ratio * num_edges)
            val_end = int((train_ratio + val_ratio) * num_edges)

            train_idx = perm[:train_end]
            val_idx = perm[train_end:val_end]
            # test_idx = perm[val_end:]

            train_data = train_val_data.clone()
            train_data.edge_label_index = train_val_data.edge_index[:, train_idx]
            train_data.edge_attr = train_val_data.edge_attr[train_idx]
            train_data.y = train_data.y[train_idx]

            val_data = train_val_data.clone()
            val_data.edge_label_index = train_val_data.edge_index[:, val_idx]
            val_data.y = train_val_data.y[val_idx]
            val_data.edge_attr = train_val_data.edge_attr[val_idx]

            test_data.num_current_edges = test_data.num_edges
            test_data.num = test_data.num_nodes
            test_data.edge_label_index = test_data.edge_index

            if model is None:
                model = EdgeHawkGNN(gnn_type=gnn_type, n_node=dataset.num_nodes, in_dim=train_val_data.x.shape[1],
                                    hid_dim=hidden_size, layers=num_layers, dropout=dropout, use_bn=use_bn,
                                    edge_attr_dim=dataset.num_edge_features)
                # model = EdgeHawkGNN(snapshot.x.shape[1], edge_attr_dim=dataset.num_edge_features,
                #                     num_layers=num_layers, dropout=dropout,
                #                     memory_size=memory_size, hidden_size=hidden_size,
                #                     gnn_type=gnn_type, enable_memory=enable_memory)
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()

                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()

                total_size = (param_size + buffer_size) / (1024 ** 2)
                print(colored(count_model_elements(model)), "green")
                print(f"Model size (parameters + buffers): {total_size:.2f} MB")
            # else:
            #     model = EdgeHawkGNN(gnn_type=gnn_type, n_node=dataset.num_nodes, in_dim=train_val_data.x.shape[1],
            #                         hid_dim=hidden_size, layers=num_layers, dropout=dropout, use_bn=use_bn,
            #                         edge_attr_dim=dataset.num_edge_features)
            lightningModule = LightningEdgeGNN(model, learning_rate=learning_rate, alpha=alpha,
                                               anomaly_loss_margin=anomaly_loss_margin, blend_factor=blend_factor)
            # experiments_dir = f"{lightning_root_dir}/{dataset_name}/{graph_window_size}/Mem{enable_memory}_{gnn_type}_F{fresh_start}/{experiment_datetime}/index_{data_index}"
            # csv_logger = CSVLogger(experiments_dir, version="")
            csv_logger.log_hyperparams(vars(args))
            print(colored(f"Time Index: {start_index}, hawk_windwo_size:{hawk_window_size}, data: {dataset_name}",
                          "yellow"))
            print(train_data)
            print(val_data)
            print(test_data)
            # Start training and testing.
            train_loader = DataLoader([train_data], batch_size=1)
            val_loader = DataLoader([val_data], batch_size=1)
            test_loader = DataLoader([test_data], batch_size=1)
            # Callbacks
            # early_stop_callback = EarlyStopping(
            #     monitor='val_avg_pr',
            #     mode='max',
            #     patience=10
            # )
            checkpoint_callback = ModelCheckpoint(
                monitor="val_avg_pr",
                mode="max",
                save_top_k=1,
                save_weights_only=True,
                dirpath=experiments_dir,
                filename="best-checkpoint"
            )
            # model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_avg_pr")
            trainer = L.Trainer(default_root_dir=experiments_dir,
                                accelerator="auto",
                                devices="auto",
                                enable_progress_bar=True,
                                logger=csv_logger,
                                max_epochs=epochs,
                                callbacks=[checkpoint_callback]
                                )
            # Visualization embedding before training
            if embedding_visualization:
                print("start visualization")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                model.eval()
                train_data = train_data.to(device)
                train_labels = train_data.y[train_data.node_mask]
                train_labels_np = train_labels.cpu().numpy()
                with torch.no_grad():
                    all_embeddings = model.get_embedding(train_data.x, train_data.edge_index)
                    train_embeddings_np = all_embeddings[train_data.node_mask].cpu().numpy()
                train_label_colors = ['blue' if label == 0 else 'red' for label in train_labels_np]
                visualize_embeddings(train_embeddings_np, train_label_colors, 0,
                                     f'{experiments_dir}/train_embeddings_not_trained.png')

            trainer.fit(lightningModule, train_loader, val_loader)
            trainer.test(lightningModule, test_loader)

            # Visualization embedding
            if embedding_visualization:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                model.eval()
                train_data = train_data.to(device)
                test_data = test_data.to(device)
                test_labels = test_data.y[test_data.node_mask]
                test_labels_np = test_labels.cpu().numpy()
                with torch.no_grad():
                    train_embeddings = model.get_embedding(train_data.x, train_data.edge_index)
                    train_embeddings_np = train_embeddings[train_data.node_mask].cpu().numpy()
                    test_embeddings = model.get_embedding(test_data.x, test_data.edge_index)
                    test_embeddings_np = test_embeddings[test_data.node_mask].cpu().numpy()
                test_label_colors = ['blue' if label == 0 else 'red' for label in test_labels_np]
                visualize_embeddings(train_embeddings_np, train_label_colors, epochs,
                                     f'{experiments_dir}/train_embedding_trained.png')
                visualize_embeddings(test_embeddings_np, test_label_colors, 'None',
                                     f'{experiments_dir}/test_embedding.png')


if __name__ == "__main__":
    main()

import argparse
import torch
from termcolor import colored
from colorama import init
import copy
import gc
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from datasets.data_loading import get_dataset
from torch_geometric.data import DataLoader
from models.dyfraudnet.model import NodeDyFraudNet, EdgeDyFraudNet
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
                        choices=["EllipticPP", "DGraphFin", "BitcoinOTC", "MOOC",
                                 "RedditTitle", "RedditBody", "EthereumPhishing", "SAMLSim",
                                 "AMLWorldLarge", "AMLWorldMedium", "AMLWorldSmall"], default="RedditTitle")
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
    for data_index in range(len(dataset) - 1):
        snapshot = dataset[data_index]
        if snapshot.x is None:
            snapshot.x = torch.Tensor([[1] for _ in range(snapshot.num_nodes)])
        if snapshot.x is None:
            snapshot.x = torch.Tensor([[1] for _ in range(snapshot.num_nodes)])
        experiments_dir = f"{lightning_root_dir}/{dataset_name}/{graph_window_size}/Mem{enable_memory}_{gnn_type}_F{fresh_start}/{experiment_datetime}/index_{data_index}"
        csv_logger = CSVLogger(experiments_dir, version="")
        if task == "Node":
            train_mask = torch.zeros_like(snapshot.node_mask, dtype=torch.bool)
            val_mask = torch.zeros_like(snapshot.node_mask, dtype=torch.bool)
            train_indices = snapshot.node_mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(train_indices))
            split_idx = int(0.8 * len(train_indices))
            train_mask[train_indices[perm[:split_idx]]] = True
            val_mask[train_indices[perm[split_idx:]]] = True
            train_data = snapshot.clone()
            train_data.node_mask = train_mask

            val_data = snapshot.clone()
            val_data.node_mask = val_mask
            test_data = copy.deepcopy(dataset[data_index + 1])
            test_data.num_current_edges = test_data.num_edges
            test_data.num = test_data.num_nodes
            if snapshot.x is None:
                test_data.x = torch.Tensor([[1] for _ in range(test_data.num_nodes)])
            if (model is None) or fresh_start:
                model = NodeDyFraudNet(snapshot.x.shape[1], memory_size=memory_size, hidden_size=hidden_size,
                                       out_put_size=2, dropout=dropout, num_layers=num_layers,
                                       gnn_type=gnn_type, enable_memory=enable_memory)
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
            csv_logger.log_hyperparams(vars(args))
            print(colored(f"Time Index: {data_index}, data: {dataset_name}", "yellow"))
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
                                max_epochs=epochs, precision="16-mixed",
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
            del train_data, train_loader, val_data, val_loader, test_data, test_loader, snapshot
            gc.collect()
            torch.cuda.empty_cache()

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

            val_data = snapshot.clone()
            val_data.edge_label_index = snapshot.edge_index[:, val_idx]
            val_data.y = snapshot.y[val_idx]
            val_data.edge_attr = snapshot.edge_attr[val_idx]

            test_data = copy.deepcopy(dataset[data_index + 1])
            test_data.num_current_edges = test_data.num_edges
            test_data.num = test_data.num_nodes
            test_data.edge_label_index = test_data.edge_index

            if snapshot.x is None:
                test_data.x = torch.Tensor([[1] for _ in range(test_data.num_nodes)])
            if (model is None) or fresh_start:
                model = EdgeDyFraudNet(snapshot.x.shape[1], edge_attr_dim=dataset.num_edge_features,
                                       num_layers=num_layers, dropout=dropout,
                                       memory_size=memory_size, hidden_size=hidden_size,
                                       gnn_type=gnn_type, enable_memory=enable_memory)
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()

                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()

                total_size = (param_size + buffer_size) / (1024 ** 2)
                print(colored(count_model_elements(model), "green"))
                print(f"Model size (parameters + buffers): {total_size:.2f} MB")
            lightningModule = LightningEdgeGNN(model, learning_rate=learning_rate, alpha=alpha,
                                               anomaly_loss_margin=anomaly_loss_margin, blend_factor=blend_factor)
            # experiments_dir = f"{lightning_root_dir}/{dataset_name}/{graph_window_size}/Mem{enable_memory}_{gnn_type}_F{fresh_start}/{experiment_datetime}/index_{data_index}"
            # csv_logger = CSVLogger(experiments_dir, version="")
            csv_logger.log_hyperparams(vars(args))
            print(colored(f"Time Index: {data_index}, data: {dataset_name}", "yellow"))
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
                                max_epochs=epochs,precision="16-mixed",
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
            del train_data, train_loader, val_data, val_loader, test_data, test_loader, snapshot
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

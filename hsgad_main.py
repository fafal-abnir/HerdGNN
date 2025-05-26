import copy
import torch
import argparse
from datetime import datetime
from datasets.data_loading import get_dataset
from utils.visualization import visualize_embeddings
from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from models.hsgad.model import NodeHSGAD
from models.hsgad.lightning_modules import LightningNodeGNN
import pytorch_lightning as L

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser(description="Semi-supervised Anomaly Detection on Attributed Graphs")
    parser.add_argument("--fresh_start", action="store_true", help="retraining from scratch on each timestamp")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 10)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number GNN layers")
    parser.add_argument("--gnn_type", type=str, choices=["GIN", "GCN", "GAT"])
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--lam", type=float, default=1.0, help="Regularize factor for AUC loss")
    parser.add_argument("--hidden_size", type=int, default=128, help="embedding size of the GNN layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate of the model")
    parser.add_argument("--dataset_name", type=str, choices=["DGraphFin"], help="Just support DGraphFin")
    parser.add_argument("--graph_window_size", type=str, choices=["day", "week", "month"], default="month",
                        help="the size of graph window size")
    parser.add_argument("--force_reload_dataset", action="store_true")
    parser.add_argument("--num_windows", type=int, default=10, help="Number of windows for running the experiment")
    parser.add_argument("--embedding_visualization", action="store_true",
                        help="Visualization of train data before and after training")
    return parser.parse_args()


def main():
    args = get_args()
    fresh_start = args.fresh_start
    num_layers = args.num_layers
    gnn_type = args.gnn_type
    learning_rate = args.learning_rate
    epochs = args.epochs
    lam = args.lam
    hidden_size = args.hidden_size
    dropout = args.dropout
    dataset_name = args.dataset_name
    graph_window_size = args.graph_window_size
    num_windows = args.num_windows
    force_reload_dataset = args.force_reload_dataset
    embedding_visualization = args.embedding_visualization
    model = None
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if dataset_name in ["DGraphFin"]:
        task = "Node"
        lightning_root_dir = "experiments/hsgad/node_level"
    else:
        task = "Edge"
        lightning_root_dir = "experiments/hsgad/edge_level"
    dataset = get_dataset(name=dataset_name, force_reload=force_reload_dataset, edge_window_size=graph_window_size,
                          num_windows=num_windows)
    print(f"Number of windows: {len(dataset)}")
    for data_index in range(len(dataset) - 1):
        snapshot = dataset[data_index]
        if snapshot.x is None:
            snapshot.x = torch.Tensor([[1] for _ in range(snapshot.num_nodes)])
        if snapshot.x is None:
            snapshot.x = torch.Tensor([[1] for _ in range(snapshot.num_nodes)])
        experiments_dir = f"{lightning_root_dir}/{dataset_name}/{graph_window_size}/{gnn_type}/{experiment_datetime}/index_{data_index}"
        csv_logger = CSVLogger(experiments_dir, version="")
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
            model = NodeHSGAD(snapshot.x.shape[1], hidden_size=hidden_size,
                              out_put_size=2, dropout=dropout, num_layers=num_layers,
                              gnn_type=gnn_type)
        lightningModule = LightningNodeGNN(model, learning_rate=learning_rate, lam=lam)
        csv_logger.log_hyperparams(vars(args))
        print(f"Time Index: {data_index}, data: {dataset_name}")
        print(train_data)
        print(val_data)
        print(test_data)
        # Start training and testing.
        train_loader = DataLoader([train_data], batch_size=1)
        val_loader = DataLoader([val_data], batch_size=1)
        test_loader = DataLoader([test_data], batch_size=1)
        trainer = L.Trainer(default_root_dir=experiments_dir,
                            accelerator="auto",
                            devices="auto",
                            enable_progress_bar=True,
                            logger=csv_logger,
                            max_epochs=epochs
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


if __name__ == "__main__":
    main()

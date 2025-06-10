import os
import pandas as pd
from typing import Callable, Optional

import torch
from sklearn.preprocessing import MinMaxScaler

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)



class EllipticPP(InMemoryDataset):
    r"""The Elliptic++ dataset for graph-based fraud detection.

    The dataset contains a dynamic transaction graph where nodes are accounts
    and edges are transactions. Each node has feature vectors and labels
    (fraud or not), and each edge is timestamped.

    Args:
        root (str): Root directory where the dataset should be saved.
        num_windows (int, optional): Number of temporal snapshots to keep.
        transform (callable, optional): Data transform function.
        pre_transform (callable, optional): Pre-processing transform function.
        force_reload (bool, optional): Whether to reprocess the dataset.
    """

    url = "https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l"

    def __init__(self,
                 root: str,
                 num_windows: int = 3,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False) -> None:
        self.num_windows = num_windows
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])
    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self):
        return ['txs_features.csv', 'txs_classes.csv', 'txs_edgelist.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return 202_804

    @property
    def num_node_features(self) -> int:
        return 183

    @property
    def num_classes(self) -> int:
        return 2

    def process(self) -> None:
        feature_path = os.path.join(self.raw_dir, "txs_features.csv")
        label_path = os.path.join(self.raw_dir, "txs_classes.csv")
        edge_list_path = os.path.join(self.raw_dir, "txs_edgelist.csv")
        feature_df = pd.read_csv(feature_path)
        feature_df = feature_df.dropna()
        scaler = MinMaxScaler()
        for column in feature_df.columns[168:]:
            values = feature_df[[column]].values  # Keep 2D shape
            scaled = scaler.fit_transform(values).ravel()
            feature_df[column] = scaled

        edge_list_df = pd.read_csv(edge_list_path)
        valid_tx_ids = set(feature_df['txId'])

        edge_list_df = edge_list_df[
            edge_list_df['txId1'].isin(valid_tx_ids) &
            edge_list_df['txId2'].isin(valid_tx_ids)
            ].reset_index(drop=True)

        label_df = pd.read_csv(label_path)
        txids = feature_df['txId'].unique()
        txid2idx = {uid: idx for idx, uid in enumerate(txids)}

        feature_df['tx_idx'] = feature_df['txId'].map(txid2idx)
        label_df['tx_idx'] = label_df['txId'].map(txid2idx)

        label_df['class'] = label_df['class'].apply(lambda va: 1 if va == 1 else 0)
        feature_df = feature_df.merge(label_df, "inner", on="tx_idx")
        feature_df.drop(columns=['txId_x', 'txId_y'], inplace=True)
        feature_df = feature_df.sort_values(by='tx_idx')
        edge_list_df['source_idx'] = edge_list_df['txId1'].map(txid2idx)
        edge_list_df['target_idx'] = edge_list_df['txId2'].map(txid2idx)
        tx_time_df = feature_df[['tx_idx', 'Time step']]
        edge_list_df = edge_list_df.merge(tx_time_df, 'inner', left_on="target_idx", right_on='tx_idx')
        edge_list_df = edge_list_df[['source_idx', 'target_idx', 'Time step']]
        features_x = feature_df.drop(columns=['tx_idx', 'Time step', 'class'])
        x = torch.tensor(features_x.values, dtype=torch.float)
        y = torch.tensor(feature_df['class'].values, dtype=torch.float)
        data_list = []
        counter = 0
        for i, group in edge_list_df.groupby('Time step'):
            data = Data()
            data.x = x
            edge_index = torch.tensor(group[['source_idx', 'target_idx']].values, dtype=torch.long)
            data.edge_index = torch.tensor(edge_index.t(), dtype=torch.long)
            data.y = y
            available_nodes = torch.unique(edge_index)
            data.num_nodes = available_nodes.size(0)
            node_mask = torch.zeros(x.shape[0], dtype=torch.bool)
            # filtered_indices = available_nodes[available_nodes[(data.y[available_nodes] == 0) | (data.y[available_nodes] == 1)]]
            node_mask[available_nodes] = True
            data.node_mask = node_mask
            data_list.append(data)
            counter += 1
            if counter >= self.num_windows:
                break

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

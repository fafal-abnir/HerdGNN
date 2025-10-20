import os.path as osp
import pandas as pd
import numpy as np
from collections import Counter
from typing import Callable, Optional, Literal
from sklearn.preprocessing import StandardScaler

import torch
from sklearn.preprocessing import MinMaxScaler

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)



class SAMLSim(InMemoryDataset):
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

    url = "www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml/data?select=SAML-D.csv"

    def __init__(self,
                 root: str,
                 num_windows: int = 3,
                 edge_window_size: Literal["day", "week", "month"] = "week",
                 node_feature_mode: Literal["degree", "one"] = "degree",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False) -> None:
        self.num_windows = num_windows
        self.edge_window_size = {"week": 'W', "day": 'D'}.get(edge_window_size, 'M')
        self.node_feature_mode = node_feature_mode
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])
    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self):
        return 'SAML-D.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return 855_460

    @property
    def num_node_features(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        return 2

    def process(self) -> None:
        path = osp.join(self.raw_dir, "SAML-D.csv")
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        df.rename(columns={
            "sender_account": "sender",
            "receiver_account": "receiver",
            "amount": "amount",
            "is_laundering": "label",
        }, inplace=True)
        df['date'] = pd.to_datetime((df['date']))
        df = df.sort_values('date')

        # Reindex sender and receiver
        all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
        num_nodes = len(all_ids)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
        df["sender"] = df["sender"].map(id_mapping)
        df["receiver"] = df["receiver"].map(id_mapping)
        cat_cols = ["payment_currency", "received_currency",
                    "sender_bank_location", "receiver_bank_location",
                    "payment_type"]
        df = pd.get_dummies(df, columns=cat_cols, dtype=int)
        df["amount"] = np.log1p(df["amount"].clip(lower=0))
        scaler = StandardScaler()
        df["amount"] = scaler.fit_transform(df[["amount"]])
        dummy_prefixes = [f"{c}_" for c in cat_cols]
        dummy_cols = [c for c in df.columns if any(c.startswith(p) for p in dummy_prefixes)]
        edge_feat_cols = dummy_cols + ['amount']
        df = df.drop(columns=['laundering_type'])
        df["bin"] = df["date"].dt.to_period(self.edge_window_size)

        data_list = []
        counter = 0
        for i, gdf in df.groupby('bin'):
            data = Data()
            # Node features
            if self.node_feature_mode == "degree":
                # degree within this snapshot (undirected count of incidences)
                deg = Counter(gdf["sender"].tolist() + gdf["receiver"].tolist())
                x_vals = torch.zeros(num_nodes, 1, dtype=torch.float32)
                if deg:
                    idxs = torch.tensor(list(deg.keys()), dtype=torch.long)
                    vals = torch.tensor(list(deg.values()), dtype=torch.float32).unsqueeze(1)
                    x_vals[idxs] = vals
                x = x_vals
            else:
                x = torch.ones(num_nodes, 1, dtype=torch.float32)
            data.x = x
            data.edge_index = torch.tensor(gdf[['sender', 'sender']].values.T, dtype=torch.long)
            data.y = torch.tensor(gdf['label'].values, dtype=torch.float)
            data.edge_attr = torch.tensor(gdf[edge_feat_cols].to_numpy(),
                                          dtype=torch.float)
            data_list.append(data)
            counter += 1
            if counter >= self.num_windows:
                break

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

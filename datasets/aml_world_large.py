import os.path as osp
import pandas as pd
import numpy as np
from collections import Counter
from typing import Callable, Optional, Literal
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from tqdm import tqdm
import torch
from sklearn.preprocessing import MinMaxScaler

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)


class AMLWorld(InMemoryDataset):
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

    url = "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data"

    def __init__(self,
                 root: str,
                 num_windows: int = 3,
                 edge_window_size: Literal["day", "week", "month"] = "day",
                 illicit_level: Literal["HI", "LI"] = "HI",
                 data_size: Literal["Small", "Medium", "Large"] = "Small",
                 node_feature_mode: Literal["degree", "one"] = "degree",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False) -> None:
        self.num_windows = num_windows
        self.edge_window_size = {"week": 'W', "day": 'D'}.get(edge_window_size, 'M')
        self.node_feature_mode = node_feature_mode
        self.csv_file_name = f"{illicit_level}-{data_size}_Trans.csv"
        root = root + "/" + f"{illicit_level}-{data_size}"
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download '{self.csv_file_name}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self):
        return self.csv_file_name

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        if self.csv_file_name == "HI-Small_Trans.csv":
            return 515_080
        else:
            return 2_076_999

    @property
    def num_node_features(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        return 2

    def process(self) -> None:
        path = osp.join(self.raw_dir, self.csv_file_name)
        df = pd.read_csv(path)
        df.rename(columns={
            "Account": "sender",
            "Account.1": "receiver",
            "Amount Received": "amount",
            "Is Laundering": "label"
        }, inplace=True)
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
        df['date_time'] = pd.to_datetime((df['timestamp']))

        df = df.sort_values('date_time')
        # Reindex sender and receiver
        all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
        num_nodes = len(all_ids)
        print(f"Number of nodes:{num_nodes} edge: {len(df)}")
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
        df["sender"] = df["sender"].map(id_mapping)
        df["receiver"] = df["receiver"].map(id_mapping)
        df["amount"] = np.log1p(df["amount"].clip(lower=0))
        scaler = StandardScaler()
        df["amount"] = scaler.fit_transform(df[["amount"]])

        # cat_cols = ['from_bank', 'to_bank', 'receiving_currency', "payment_currency", "payment_format"]
        # df = pd.get_dummies(df, columns=cat_cols, dtype=int)
        # dummy_prefixes = [f"{c}_" for c in cat_cols]
        # dummy_cols = [c for c in df.columns if any(c.startswith(p) for p in dummy_prefixes)]
        # edge_feat_cols = dummy_cols + ['amount']
        cat_cols = ['from_bank', 'to_bank', 'receiving_currency',
                    'payment_currency', 'payment_format']
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols])
        edge_feat_cols = cat_cols + ['amount']

        df["edge_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
        # ordered = [
        #     "edge_id",  # 0
        #     "sender",  # 1
        #     "receiver",  # 2
        #     "timestamp",  # 3
        #     "from_bank",  # 4
        #     "to_bank",  # 5
        #     "amount",  # 6
        #     "receiving_currency",  # 7
        #     "payment_currency",  # 8
        #     "payment_format",  # 9
        #     "label",  # 10 (drop before transform)
        # ]
        # extra = [c for c in df.columns if c not in ordered]
        # df = df[ordered + extra]
        df["bin"] = df["date_time"].dt.to_period(self.edge_window_size)
        data_list = []
        counter = 0
        for i, gdf in tqdm(df.groupby('bin')):
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

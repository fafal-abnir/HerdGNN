import datetime
import os
import numpy as np
import pandas as pd
from typing import Literal
from typing import Callable, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)


class BitcoinOTC(InMemoryDataset):
    url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'

    def __init__(
            self,
            root: str,
            edge_window_size: Literal["day", "week", "month"] = "week",
            num_windows: int = 50,
            fraud_threshold=0.1,
            benign_threshold=0.6,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = True,
    ) -> None:
        self.edge_window_size = {"week": 7, "day": 1}.get(edge_window_size, 30)
        self.num_windows = num_windows
        self.fraud_threshold = fraud_threshold
        self.benign_threshold = benign_threshold
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'soc-sign-bitcoinotc.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return 5_881

    @property
    def num_node_features(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        return 2

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        path = os.path.join(self.raw_dir, "soc-sign-bitcoinotc.csv")
        bitcoin_df = pd.read_csv(path, names=['src', 'dest', 'rating', 'timestamp'])
        # preparing data
        user_ids = pd.unique(bitcoin_df[['src', 'dest']].values.ravel())
        id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        bitcoin_df['src_id'] = bitcoin_df['src'].map(id_to_index)
        bitcoin_df['dest_id'] = bitcoin_df['dest'].map(id_to_index)
        num_nodes = len(id_to_index)
        # preparing node labels
        incoming = bitcoin_df.groupby('dest_id').agg(
            total_in_rating=('rating', 'sum'),
            num_in_rating=('rating', 'count')
        ).reset_index()
        incoming['avg_in_rating'] = incoming['total_in_rating'] / incoming['num_in_rating']
        global_avg_rating = bitcoin_df['rating'].sum() / bitcoin_df['dest_id'].nunique()
        conditions = [
            incoming['avg_in_rating'] < self.fraud_threshold,
            incoming['avg_in_rating'] > (global_avg_rating - self.benign_threshold),
        ]
        choices = [1, 0]  # fraud, benign
        incoming['label'] = np.select(conditions, choices, default=0)  # 2 unknown
        label_series = pd.Series(2, index=range(num_nodes))
        label_series.update(incoming['label'])
        y = torch.tensor(label_series.values, dtype=torch.long)

        #
        bitcoin_df['datetime'] = pd.to_datetime(bitcoin_df['timestamp'], unit='s')
        if self.edge_window_size == "day":
            bitcoin_df['date'] = bitcoin_df['datetime'].dt.date
        elif self.edge_window_size == "week":
            bitcoin_df['date'] = bitcoin_df['datetime'].dt.to_period('W').dt.start_time
        else:
            bitcoin_df['date'] = bitcoin_df['datetime'].dt.to_period('M').dt.start_time
        data_list = []
        counter = 0
        for week, group in bitcoin_df.groupby('date'):
            edge_index = torch.tensor([group['src_id'].values, group['dest_id'].values], dtype=torch.long)
            edge_attr = torch.tensor(group['rating'].values, dtype=torch.float).unsqueeze(1)
            data = Data()
            x = torch.ones((num_nodes, 1), dtype=torch.float)
            data.x = x
            data.y = y
            available_nodes = torch.unique(edge_index.flatten())
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            filtered_indices = available_nodes[(data.y[available_nodes] == 0) | (data.y[available_nodes] == 1)]
            node_mask[filtered_indices] = True
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            data.num_nodes = len(filtered_indices)
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

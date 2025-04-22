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
    extract_tar,
)


class MOOC(InMemoryDataset):
    url = 'https://snap.stanford.edu/data/act-mooc.tar.gz'

    def __init__(
            self,
            root: str,
            edge_window_size: Literal["day", "week", "month"] = "day",
            num_windows: int = 30,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = True,
    ) -> None:
        self.edge_window_size = {"week": 7, "day": 1}.get(edge_window_size, 30)
        self.num_windows = num_windows
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['mooc_action_features.tsv', 'mooc_action_labels.tsv', 'mooc_actions.tsv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return 7_144

    @property
    def num_node_features(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        return 2

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        # ['mooc_action_features.tsv', 'mooc_action_labels.tsv', 'mooc_actions.tsv']
        # path = os.path.join(self.raw_dir, "soc-sign-bitcoinotc.csv")
        feature_path = os.path.join(self.raw_dir,"act-mooc/mooc_action_features.tsv")
        label_path = os.path.join(self.raw_dir, "act-mooc/mooc_action_labels.tsv")
        actions_path = os.path.join(self.raw_dir, "act-mooc/mooc_actions.tsv")
        df_action_features = pd.read_csv(feature_path, sep="\t")
        df_action_labels = pd.read_csv(label_path, sep="\t")
        df_actions = pd.read_csv(actions_path, sep="\t")
        df_actions_merged = pd.merge(df_actions, df_action_features, on="ACTIONID")
        df_actions_merged = pd.merge(df_actions_merged, df_action_labels, on="ACTIONID")
        df_actions_merged.columns = df_actions_merged.columns.str.lower()
        # re-indexing id for safety
        user_ids = df_actions_merged['userid'].unique()
        target_ids = df_actions_merged['targetid'].unique()
        user_id2idx = {uid: idx for idx, uid in enumerate(user_ids)}
        target_id2idx = {tid: idx + len(user_ids) for idx, tid in enumerate(target_ids)}
        df_actions_merged['source_idx'] = df_actions_merged['userid'].map(user_id2idx)
        df_actions_merged['target_idx'] = df_actions_merged['targetid'].map(target_id2idx)
        # day_group
        df_actions_merged["datetime"] = pd.to_datetime(df_actions_merged["timestamp"], unit='s')
        start = df_actions_merged["datetime"].min()
        df_actions_merged["day_group"] = ((df_actions_merged["datetime"] - start).dt.total_seconds() // 86400).astype(
            int)

        num_nodes = len(user_id2idx) + len(target_id2idx)
        data_list = []
        counter = 0
        for week, group in df_actions_merged.groupby('day_group'):
            data = Data()
            data.x = torch.ones((num_nodes, 1), dtype=torch.float)
            data.edge_index = torch.tensor(group[['source_idx', 'target_idx']].values.T, dtype=torch.long)
            data.y = torch.tensor(group['label'].values, dtype=torch.float)
            data.edge_attr = torch.tensor(group[['feature0', 'feature1', 'feature2', 'feature3']].values,
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

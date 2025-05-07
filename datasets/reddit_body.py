import os
import numpy as np
import pandas as pd
from typing import Literal, Callable, Optional
import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url
)


def random_embedding():
    return ((np.random.rand(300) * 1e5).astype(int) / 1e5).tolist()


def parse_feature_string(feat_str):
    return [float(f) for f in feat_str.split(',')]


class RedditBody(InMemoryDataset):
    urls = ['https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv',
            'https://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv']

    def __init__(self,
                 root: str,
                 edge_window_size: Literal["data", "week", "month"] = "week",
                 num_windows: int = 50,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = True
                 ) -> None:
        self.edge_window_size = {"week": 7, "day": 1}.get(edge_window_size, 30)
        self.num_windows = num_windows
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['soc-redditHyperlinks-body.tsv', 'web-redditEmbeddings-subreddits.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return 35_776
    @property
    def num_node_features(self) -> int:
        return 300

    @property
    def num_classes(self) -> int:
        return 2

    def download(self) -> None:
        download_url(self.urls[0], self.raw_dir)
        download_url(self.urls[1], self.raw_dir)

    def process(self) -> None:
        body_path = os.path.join(self.raw_dir, "soc-redditHyperlinks-body.tsv")
        embedding_path = os.path.join(self.raw_dir, "web-redditEmbeddings-subreddits.csv")
        body_df = pd.read_csv(body_path, sep="\t")
        subreddit_embedding_df = pd.read_csv(embedding_path, header=None)
        embedding_cols = [col for col in subreddit_embedding_df.columns[1:]]
        subreddit_embedding_df['embedding'] = subreddit_embedding_df[embedding_cols].values.tolist()
        subreddit_embedding_df = subreddit_embedding_df.drop(columns=embedding_cols)
        subreddit_embedding_df.columns.values[0] = 'node_id'

        body_subreddit_name_df = pd.DataFrame(
            pd.unique(body_df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].values.ravel()), columns=['node_id'])
        node_embeddings = body_subreddit_name_df.merge(subreddit_embedding_df,
                                                       left_on='node_id',
                                                       right_on='node_id',
                                                       how='left')

        node_embeddings['embedding'] = node_embeddings['embedding'].apply(
            lambda x: random_embedding() if isinstance(x, float) and pd.isna(x) else x)
        node_id_to_index = {nid: i for i, nid in enumerate(node_embeddings['node_id'])}
        node_embeddings['node_index'] = node_embeddings['node_id'].map(node_id_to_index)
        body_df['source_idx'] = body_df['SOURCE_SUBREDDIT'].map(node_id_to_index)
        body_df['target_idx'] = body_df['TARGET_SUBREDDIT'].map(node_id_to_index)
        body_df['label'] = body_df['LINK_SENTIMENT'].map({-1: 1, 1: 0})
        body_df['edge_features'] = body_df['PROPERTIES'].apply(parse_feature_string)
        body_df["datetime"] = pd.to_datetime(body_df["TIMESTAMP"])
        if self.edge_window_size == 1:
            body_df['time_group'] = body_df['datetime'].dt.to_period('D')
        elif self.edge_window_size == 7:
            body_df['time_group'] = body_df['datetime'].dt.to_period('W')
        else:
            body_df['time_group'] = body_df['datetime'].dt.to_period('M')
        grouped_body_df = body_df.groupby('time_group')
        x = torch.tensor(node_embeddings['embedding'].tolist(), dtype=torch.float)
        data_list = []
        counter = 0
        for _, group in grouped_body_df:
            data = Data()
            data.x = x
            data.edge_index = torch.tensor(group[['source_idx', 'target_idx']].values.T, dtype=torch.long)
            data.y = torch.tensor(group['label'].values, dtype=torch.float)
            data.edge_attr = torch.tensor(group['edge_features'].tolist(),
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

import os.path as osp
from typing import Callable, Optional, Literal
import pickle
import numpy as np
import torch
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset, extract_zip


class EthereumPhishing(InMemoryDataset):
    url1 = "https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network"
    url2 = "https://xblock.pro/ethereum#/search?types=datasets&tags=Transaction+Analysis"

    def __init__(self,
                 root: str,
                 edge_window_size: Literal["day", "week", "month"] = "month",
                 num_windows: int = 3,
                 node_feature_mode: Literal["degree", "one"] = "degree",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False
                 ) -> None:
        self.edge_window_size = {"week": 'W', "day": 'D'}.get(edge_window_size, 'M')
        self.num_windows = num_windows
        self.node_feature_mode = node_feature_mode
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url1}' or '{self.url2}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self) -> str:
        return 'MulDiGraph.pkl'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return 2_973_489

    @property
    def num_node_features(self) -> int:
        return 1

    @property
    def num_classes(self) -> int:
        return 2

    def process(self) -> None:
        path = osp.join(self.raw_dir, "MulDiGraph.pkl")
        with open(path, 'rb') as f:
            G = pickle.load(f)
        cutoff = pd.Timestamp("2017-07-01", tz="UTC")
        cutoff_s = int(cutoff.timestamp())

        edges_iter = G.edges(keys=True, data=True)
        ef = []
        append = ef.append
        get = dict.get

        for u, v, k, attrs in edges_iter:
            ts = get(attrs, "timestamp")
            if ts is None:
                continue
            try:
                ts_i = ts if isinstance(ts, (int, float)) else int(ts)
            except (TypeError, ValueError):
                continue
            if ts_i >= cutoff_s:
                append((u, v, k, attrs))
        print("Filtering edge is done.")
        edges_filtered = ef
        nodes = {u for u, v, _, _ in edges_filtered} | {v for u, v, _, _ in edges_filtered}
        nodes = list(nodes)
        node2idx = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)

        n2i = node2idx
        get = dict.get
        data = [
            (n2i[u], n2i[v], int(attrs["timestamp"]), float(get(attrs, "amount", 0.0)))
            for (u, v, _k, attrs) in edges_filtered
        ]
        arr = np.asarray(data, dtype=np.float64)  # fast to construct; cast ints after
        df = pd.DataFrame(arr, columns=["u", "v", "timestamp", "amount"])
        df[["u", "v", "timestamp"]] = df[["u", "v", "timestamp"]].astype("int64")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        print("Preparing dataframe for generating a pyg data list")
        if df.empty:
            raise ValueError("No edges with timestamps found.")
        df["amount"] = np.log1p(df["amount"])
        df["bin"] = df["timestamp"].dt.to_period(self.edge_window_size)
        y = torch.zeros(num_nodes, dtype=torch.long)
        for n, attrs in G.nodes(data=True):
            if n not in node2idx:
                continue
            val = attrs.get('isp', 0)
            try:
                y[node2idx[n]] = int(val)
            except Exception:
                y[node2idx[n]] = 1 if bool(val) else 0
        num_anomalies = (y == 1).sum().item()
        ratio = num_anomalies / len(y) if len(y) > 0 else 0.0
        print(f"Anomaly ratio: {num_anomalies}/{len(y)} = {ratio:.5f}")
        data_list = []
        for period, gdf in df.groupby("bin", sort=True):
            src = torch.tensor(gdf["u"].to_numpy(), dtype=torch.long)
            dst = torch.tensor(gdf["v"].to_numpy(), dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
            edge_attr = torch.tensor(gdf["amount"].to_numpy(), dtype=torch.float32).unsqueeze(1)
            # Edge attributes (amount)
            edge_attr = torch.tensor(gdf["amount"].to_numpy(), dtype=torch.float32).unsqueeze(1)

            # Node features
            if self.node_feature_mode == "degree":
                # degree within this snapshot (undirected count of incidences)
                deg = Counter(gdf["u"].tolist() + gdf["v"].tolist())
                x_vals = torch.zeros(num_nodes, 1, dtype=torch.float32)
                if deg:
                    idxs = torch.tensor(list(deg.keys()), dtype=torch.long)
                    vals = torch.tensor(list(deg.values()), dtype=torch.float32).unsqueeze(1)
                    x_vals[idxs] = vals
                x = x_vals
            else:
                x = torch.ones(num_nodes, 1, dtype=torch.float32)
            present_nodes = torch.zeros(num_nodes, dtype=torch.bool)
            active_nodes = torch.unique(torch.cat([src, dst], dim=0), sorted=True)
            num_active = int(active_nodes.numel())
            if len(src) > 0:
                present_nodes[src] = True
                present_nodes[dst] = True
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,  # node labels
                node_mask=present_nodes,
                num_nodes=num_active,
            )
            print(data)
            count_ones = (y[present_nodes] == 1).sum().item()
            print(count_ones)
            data_list.append(data)
            if len(data_list) >= self.num_windows:
                break
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        self.save(data_list, self.processed_paths[0])

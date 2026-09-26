import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):

    def __init__(
        self,
        df,
        nodes,
        features,
        lookback=12,
        horizon=1
    ):

        self.lookback = lookback
        self.horizon = horizon

        self.nodes = nodes
        self.features = features

        # Make a complete timestamp × road matrix
        timestamps = sorted(
            df["timestamp"].unique()
        )

        self.timestamps = timestamps

        node_to_index = {
            node: i
            for i, node in enumerate(nodes)
        }

        time_to_index = {
            time: i
            for i, time in enumerate(timestamps)
        }

        n_times = len(timestamps)
        n_nodes = len(nodes)
        n_features = len(features)

        data = np.zeros(
            (
                n_times,
                n_nodes,
                n_features
            ),
            dtype=np.float32
        )

        mask = np.zeros(
            (
                n_times,
                n_nodes
            ),
            dtype=bool
        )

        for _, row in df.iterrows():

            t = time_to_index[
                row["timestamp"]
            ]

            node = node_to_index[
                row["road_segment_id"]
            ]

            data[
                t,
                node,
                :
            ] = row[features].values

            mask[
                t,
                node
            ] = True

        # Fill missing observations
        for t in range(n_times):

            for node in range(n_nodes):

                if not mask[t, node]:

                    if t > 0:

                        data[t, node] = data[
                            t - 1,
                            node
                        ]

                    else:

                        data[t, node] = 0

        self.data = data

        self.samples = []

        max_start = (
            n_times
            - lookback
            - horizon
            + 1
        )

        for start in range(max_start):

            input_end = start + lookback

            target_end = (
                input_end + horizon
            )

            self.samples.append(
                (
                    start,
                    input_end,
                    target_end
                )
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        start, input_end, target_end = \
            self.samples[idx]

        X = self.data[
            start:input_end
        ]

        y = self.data[
            input_end:target_end,
            :,
            0
        ]

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return X, y
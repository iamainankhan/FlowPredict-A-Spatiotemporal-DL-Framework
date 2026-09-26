import numpy as np
import pandas as pd


def create_speed_correlation_graph(
    train_df,
    threshold=0.60
):

    print("\nCreating spatial graph...")

    # Pivot:
    # rows = timestamp
    # columns = road segment
    # values = speed

    speed_matrix = train_df.pivot_table(
        index="timestamp",
        columns="road_segment_id",
        values="avg_speed_kmph",
        aggfunc="mean"
    )

    # Fill missing values
    speed_matrix = speed_matrix.interpolate(
        method="linear"
    )

    speed_matrix = speed_matrix.ffill().bfill()

    # Correlation between road segments
    correlation = speed_matrix.corr()

    nodes = list(correlation.columns)

    n_nodes = len(nodes)

    adjacency = np.zeros(
        (n_nodes, n_nodes),
        dtype=np.float32
    )

    for i in range(n_nodes):

        for j in range(n_nodes):

            if i == j:

                adjacency[i, j] = 1.0

            elif correlation.iloc[i, j] >= threshold:

                adjacency[i, j] = correlation.iloc[i, j]

    print("Number of road segments:", n_nodes)

    print(
        "Number of spatial connections:",
        np.sum(adjacency > 0) - n_nodes
    )

    return adjacency, nodes, correlation


def normalize_adjacency(adjacency):

    # Add self connections
    A = adjacency + np.eye(
        adjacency.shape[0]
    )

    degree = np.sum(A, axis=1)

    degree_inv_sqrt = np.power(
        degree,
        -0.5
    )

    degree_inv_sqrt[
        np.isinf(degree_inv_sqrt)
    ] = 0

    D = np.diag(
        degree_inv_sqrt
    )

    normalized = D @ A @ D

    return normalized.astype(
        np.float32
    )
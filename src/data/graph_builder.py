"""Transaction graph construction for fraud detection."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class TransactionGraphBuilder:
    """Builds transaction graphs from tabular data. Users and merchants become
    nodes, transactions become edges."""

    def __init__(self, user_col="user_id", merchant_col="merchant_id",
                 amount_col="amount", timestamp_col="timestamp", label_col="is_fraud"):
        self.user_col = user_col
        self.merchant_col = merchant_col
        self.amount_col = amount_col
        self.timestamp_col = timestamp_col
        self.label_col = label_col

        self.user_mapping = {}
        self.merchant_mapping = {}

    def build_graph(self, transactions, include_features=True):
        """Build a homogeneous transaction graph."""
        # Create node mappings
        users = transactions[self.user_col].unique()
        merchants = transactions[self.merchant_col].unique()

        self.user_mapping = {u: i for i, u in enumerate(users)}
        self.merchant_mapping = {m: i + len(users) for i, m in enumerate(merchants)}

        num_nodes = len(users) + len(merchants)

        # Build edge index (user -> merchant)
        # TODO: add support for edge weights based on transaction frequency
        edge_src = transactions[self.user_col].map(self.user_mapping).values
        edge_dst = transactions[self.merchant_col].map(self.merchant_mapping).values

        # Bidirectional edges
        edge_index = torch.tensor(
            np.vstack(
                [
                    np.concatenate([edge_src, edge_dst]),
                    np.concatenate([edge_dst, edge_src]),
                ]
            ),
            dtype=torch.long,
        )

        # Edge features (amount, time features)
        edge_features = self._compute_edge_features(transactions)
        # Duplicate for bidirectional
        edge_attr = torch.cat([edge_features, edge_features], dim=0)

        # Node features
        if include_features:
            node_features = self._compute_node_features(transactions, num_nodes)
        else:
            node_features = torch.eye(num_nodes)

        # Labels (on edges, i.e., transactions)
        if self.label_col in transactions.columns:
            labels = torch.tensor(transactions[self.label_col].values, dtype=torch.long)
        else:
            labels = None

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            num_nodes=num_nodes,
        )

        # Store metadata
        data.num_users = len(users)
        data.num_merchants = len(merchants)
        data.num_transactions = len(transactions)

        return data

    def _compute_edge_features(self, transactions):
        """Compute edge (transaction) features."""
        features = []

        # Amount (normalized)
        amounts = transactions[self.amount_col].values
        amounts_norm = (amounts - amounts.mean()) / (amounts.std() + 1e-8)
        features.append(amounts_norm)

        # Log amount
        log_amounts = np.log1p(amounts)
        features.append((log_amounts - log_amounts.mean()) / (log_amounts.std() + 1e-8))

        # Time features
        if self.timestamp_col in transactions.columns:
            ts = pd.to_datetime(transactions[self.timestamp_col])

            # Hour of day (cyclical encoding)
            hour = ts.dt.hour
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))

            # Day of week
            dow = ts.dt.dayofweek
            features.append(np.sin(2 * np.pi * dow / 7))
            features.append(np.cos(2 * np.pi * dow / 7))

            # Is weekend
            features.append((dow >= 5).astype(float))

        return torch.tensor(np.column_stack(features), dtype=torch.float32)

    def _compute_node_features(self, transactions, num_nodes):
        """Compute node features for the homogeneous graph."""
        features = torch.zeros(num_nodes, 8)

        # User features
        user_stats = (
            transactions.groupby(self.user_col)
            .agg(
                {
                    self.amount_col: ["count", "mean", "std", "max"],
                    self.merchant_col: "nunique",
                }
            )
            .reset_index()
        )
        user_stats.columns = [
            "user_id",
            "tx_count",
            "avg_amount",
            "std_amount",
            "max_amount",
            "unique_merchants",
        ]

        for _, row in user_stats.iterrows():
            idx = self.user_mapping.get(row["user_id"])
            if idx is not None:
                features[idx, 0] = row["tx_count"]
                features[idx, 1] = row["avg_amount"]
                features[idx, 2] = (
                    row["std_amount"] if not np.isnan(row["std_amount"]) else 0
                )
                features[idx, 3] = row["max_amount"]
                features[idx, 4] = row["unique_merchants"]

        # Merchant features
        merchant_stats = (
            transactions.groupby(self.merchant_col)
            .agg(
                {
                    self.amount_col: ["count", "mean", "std"],
                    self.user_col: "nunique",
                }
            )
            .reset_index()
        )
        merchant_stats.columns = [
            "merchant_id",
            "tx_count",
            "avg_amount",
            "std_amount",
            "unique_users",
        ]

        for _, row in merchant_stats.iterrows():
            idx = self.merchant_mapping.get(row["merchant_id"])
            if idx is not None:
                features[idx, 5] = row["tx_count"]
                features[idx, 6] = row["avg_amount"]
                features[idx, 7] = row["unique_users"]

        # Normalize
        for i in range(features.shape[1]):
            col = features[:, i]
            if col.std() > 0:
                features[:, i] = (col - col.mean()) / col.std()

        return features

    def get_train_test_masks(self, num_samples, train_ratio=0.7, val_ratio=0.15, seed=42):
        """Create train/val/test masks."""
        np.random.seed(seed)
        indices = np.random.permutation(num_samples)

        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * val_ratio)

        train_idx = indices[:train_size]
        val_idx = indices[train_size : train_size + val_size]
        test_idx = indices[train_size + val_size :]

        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask = torch.zeros(num_samples, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask


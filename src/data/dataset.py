"""PyG dataset wrapper for fraud detection graphs."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset

from .graph_builder import TransactionGraphBuilder


class FraudDataset(InMemoryDataset):
    """Wraps transaction data into a PyG InMemoryDataset."""

    def __init__(
        self,
        root,
        transactions=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.transactions = transactions
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["transactions.csv"]

    @property
    def processed_file_names(self):
        return ["fraud_graph.pt"]

    def download(self):
        # Data should be provided externally
        pass

    def process(self):
        if self.transactions is None:
            # Try to load from raw
            raw_path = Path(self.raw_dir) / "transactions.csv"
            if raw_path.exists():
                self.transactions = pd.read_csv(raw_path)
            else:
                raise ValueError("No transaction data provided")

        # Build graph
        builder = TransactionGraphBuilder()
        data = builder.build_graph(self.transactions)

        # Create train/val/test masks
        num_edges = data.edge_index.shape[1] // 2  # Bidirectional
        train_mask, val_mask, test_mask = builder.get_train_test_masks(num_edges)

        # Expand masks for bidirectional edges
        data.train_mask = torch.cat([train_mask, train_mask])
        data.val_mask = torch.cat([val_mask, val_mask])
        data.test_mask = torch.cat([test_mask, test_mask])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def create_synthetic_fraud_data(
    num_users=1000, num_merchants=200, num_transactions=10000, fraud_rate=0.05, seed=42
):
    """Generate fake transaction data for testing."""
    np.random.seed(seed)

    # Generate base transactions
    users = [f"U{i:05d}" for i in range(num_users)]
    merchants = [f"M{i:04d}" for i in range(num_merchants)]

    data = {
        "transaction_id": [f"T{i:08d}" for i in range(num_transactions)],
        "user_id": np.random.choice(users, num_transactions),
        "merchant_id": np.random.choice(merchants, num_transactions),
        "amount": np.random.lognormal(4, 1, num_transactions),  # Log-normal amounts
        "timestamp": pd.date_range("2024-01-01", periods=num_transactions, freq="5min"),
        "device_id": [
            f"D{np.random.randint(0, 100):04d}" for _ in range(num_transactions)
        ],
    }

    df = pd.DataFrame(data)

    # Generate fraud labels
    num_fraud = int(num_transactions * fraud_rate)
    fraud_indices = np.random.choice(num_transactions, num_fraud, replace=False)

    df["is_fraud"] = 0
    df.loc[fraud_indices, "is_fraud"] = 1

    # Make fraud transactions have suspicious patterns
    # Higher amounts for fraud
    df.loc[fraud_indices, "amount"] = df.loc[fraud_indices, "amount"] * 3

    # Concentrate fraud in certain hours (night)
    fraud_timestamps = pd.to_datetime(df.loc[fraud_indices, "timestamp"])
    night_hours = np.random.randint(0, 6, len(fraud_indices))
    new_timestamps = fraud_timestamps.dt.floor("D") + pd.to_timedelta(
        night_hours, unit="h"
    )
    df.loc[fraud_indices, "timestamp"] = new_timestamps.values

    return df


def load_kaggle_fraud_data(path):
    """Load the Kaggle credit card fraud dataset and add pseudo user/merchant IDs."""
    df = pd.read_csv(path)

    # Create pseudo user/merchant IDs based on PCA features
    df["user_id"] = pd.qcut(df["V1"], q=100, labels=False, duplicates="drop").astype(
        str
    )
    df["merchant_id"] = pd.qcut(
        df["V14"], q=50, labels=False, duplicates="drop"
    ).astype(str)

    # Rename columns
    df = df.rename(
        columns={
            "Time": "timestamp",
            "Amount": "amount",
            "Class": "is_fraud",
        }
    )

    # Convert time to datetime
    df["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        df["timestamp"], unit="s"
    )

    return df[["user_id", "merchant_id", "amount", "timestamp", "is_fraud"]]


def split_temporal(
    transactions, timestamp_col="timestamp", train_ratio=0.7, val_ratio=0.15
):
    """Split by time so we don't leak future data into training."""
    df = transactions.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df

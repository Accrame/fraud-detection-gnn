"""Feature engineering for fraud detection."""

import numpy as np
import pandas as pd


class FeatureExtractor:
    """Extracts behavioral, temporal and network features from transaction data."""

    def __init__(
        self,
        user_col="user_id",
        merchant_col="merchant_id",
        amount_col="amount",
        timestamp_col="timestamp",
    ):
        self.user_col = user_col
        self.merchant_col = merchant_col
        self.amount_col = amount_col
        self.timestamp_col = timestamp_col

        self.user_stats = {}
        self.merchant_stats = {}

    def fit(self, transactions):
        """Compute aggregate stats from historical transactions."""
        # User statistics
        user_agg = transactions.groupby(self.user_col).agg(
            {
                self.amount_col: ["count", "mean", "std", "max", "min"],
                self.merchant_col: "nunique",
            }
        )
        user_agg.columns = ["_".join(col) for col in user_agg.columns]
        self.user_stats = user_agg.to_dict("index")

        # Merchant statistics
        merchant_agg = transactions.groupby(self.merchant_col).agg(
            {
                self.amount_col: ["count", "mean", "std", "max"],
                self.user_col: "nunique",
            }
        )
        merchant_agg.columns = ["_".join(col) for col in merchant_agg.columns]
        self.merchant_stats = merchant_agg.to_dict("index")

        return self

    def transform(self, transactions):
        """Extract features for transactions."""
        features = transactions.copy()

        # Amount features
        features = self._add_amount_features(features)

        # Temporal features
        features = self._add_temporal_features(features)

        # User behavioral features
        features = self._add_user_features(features)

        # Merchant features
        features = self._add_merchant_features(features)

        # Deviation features
        features = self._add_deviation_features(features)

        return features

    def fit_transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(transactions).transform(transactions)

    def _add_amount_features(self, df):
        """Amount-based features."""
        df = df.copy()

        # Log amount
        df["log_amount"] = np.log1p(df[self.amount_col])

        # Amount bins
        df["amount_bin"] = pd.qcut(
            df[self.amount_col], q=10, labels=False, duplicates="drop"
        )

        # Round amount indicator (potential structuring)
        df["is_round_amount"] = (df[self.amount_col] % 100 == 0).astype(int)

        return df

    def _add_temporal_features(self, df):
        """Time-based features."""
        df = df.copy()

        if self.timestamp_col in df.columns:
            # FIXME: this breaks if timestamp column has mixed formats
            ts = pd.to_datetime(df[self.timestamp_col])

            # Hour of day
            df["hour"] = ts.dt.hour
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

            # Day of week
            df["day_of_week"] = ts.dt.dayofweek
            df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

            # Weekend indicator
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

            # Night transaction (risky hours)
            df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)

            # Day of month
            df["day_of_month"] = ts.dt.day

        return df

    def _add_user_features(self, df):
        df = df.copy()

        # Map user statistics
        if self.user_stats:
            df["user_tx_count"] = df[self.user_col].map(
                lambda x: self.user_stats.get(x, {}).get("amount_count", 0)
            )
            df["user_avg_amount"] = df[self.user_col].map(
                lambda x: self.user_stats.get(x, {}).get("amount_mean", 0)
            )
            df["user_std_amount"] = df[self.user_col].map(
                lambda x: self.user_stats.get(x, {}).get("amount_std", 0)
            )
            df["user_max_amount"] = df[self.user_col].map(
                lambda x: self.user_stats.get(x, {}).get("amount_max", 0)
            )
            df["user_unique_merchants"] = df[self.user_col].map(
                lambda x: self.user_stats.get(x, {}).get("merchant_id_nunique", 0)
            )

            # Is new user
            df["is_new_user"] = (df["user_tx_count"] == 0).astype(int)

        return df

    def _add_merchant_features(self, df):
        df = df.copy()

        if self.merchant_stats:
            df["merchant_tx_count"] = df[self.merchant_col].map(
                lambda x: self.merchant_stats.get(x, {}).get("amount_count", 0)
            )
            df["merchant_avg_amount"] = df[self.merchant_col].map(
                lambda x: self.merchant_stats.get(x, {}).get("amount_mean", 0)
            )
            df["merchant_unique_users"] = df[self.merchant_col].map(
                lambda x: self.merchant_stats.get(x, {}).get("user_id_nunique", 0)
            )

            # Is new merchant
            df["is_new_merchant"] = (df["merchant_tx_count"] == 0).astype(int)

        return df

    def _add_deviation_features(self, df):
        """How far is this transaction from the user's normal behavior?"""
        df = df.copy()

        if self.user_stats:
            # Amount deviation from user average
            user_avg = df[self.user_col].map(
                lambda x: self.user_stats.get(x, {}).get(
                    "amount_mean", df[self.amount_col].mean()
                )
            )
            user_std = df[self.user_col].map(
                lambda x: self.user_stats.get(x, {}).get(
                    "amount_std", df[self.amount_col].std()
                )
            )

            df["amount_zscore"] = (df[self.amount_col] - user_avg) / (user_std + 1e-8)

            # Amount ratio to user max
            user_max = df[self.user_col].map(
                lambda x: self.user_stats.get(x, {}).get(
                    "amount_max", df[self.amount_col].max()
                )
            )
            df["amount_to_max_ratio"] = df[self.amount_col] / (user_max + 1e-8)

        return df

    def get_feature_names(self):
        return [
            "log_amount",
            "amount_bin",
            "is_round_amount",
            "hour",
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "is_night",
            "day_of_month",
            "user_tx_count",
            "user_avg_amount",
            "user_std_amount",
            "user_max_amount",
            "user_unique_merchants",
            "is_new_user",
            "merchant_tx_count",
            "merchant_avg_amount",
            "merchant_unique_users",
            "is_new_merchant",
            "amount_zscore",
            "amount_to_max_ratio",
        ]


def compute_velocity_features(
    transactions, user_col="user_id", timestamp_col="timestamp", windows=[1, 6, 24]
):
    """Transaction frequency in rolling time windows."""
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([user_col, "timestamp"])

    for window in windows:
        window_td = pd.Timedelta(hours=window)
        col_name = f"tx_count_{window}h"

        # Count transactions in window
        df[col_name] = (
            df.groupby(user_col)
            .apply(lambda x: x.rolling(window_td, on="timestamp")["timestamp"].count())
            .reset_index(level=0, drop=True)
        )

    return df


def compute_graph_features(
    transactions, user_col="user_id", merchant_col="merchant_id"
):
    """Graph-based features using NetworkX (degree, clustering coeff)."""
    import networkx as nx

    # Build bipartite graph
    G = nx.Graph()

    users = transactions[user_col].unique()
    merchants = transactions[merchant_col].unique()

    G.add_nodes_from(users, bipartite=0)
    G.add_nodes_from(merchants, bipartite=1)

    edges = transactions[[user_col, merchant_col]].values
    G.add_edges_from(edges)

    # Compute features
    user_degree = {node: G.degree(node) for node in users}
    merchant_degree = {node: G.degree(node) for node in merchants}

    # Clustering coefficient (for one-mode projection)
    user_projection = nx.bipartite.projected_graph(G, users)
    user_clustering = nx.clustering(user_projection)

    df = transactions.copy()
    df["user_degree"] = df[user_col].map(user_degree)
    df["merchant_degree"] = df[merchant_col].map(merchant_degree)
    df["user_clustering"] = df[user_col].map(lambda x: user_clustering.get(x, 0))

    return df

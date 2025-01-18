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
        # TODO: implement this
        raise NotImplementedError

"""GraphSAGE model for fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGraphSAGE(nn.Module):
    """GraphSAGE for node-level fraud detection."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, aggregator="mean"):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # SAGE convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggregator)
            )
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))

        # Batch normalization
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            # tried F.elu here — marginal improvement but slower
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = self.classifier(x)

        return x

    def get_embeddings(self, x, edge_index):
        """Node embeddings before the classification head."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index)
        return x

"""GAT (Graph Attention Network) for fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FraudGAT(nn.Module):
    """Multi-head attention GNN. The attention weights are useful
    for interpretability — you can see which neighbors matter most."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, heads=4, dropout=0.3, attention_dropout=0.3):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GAT layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=attention_dropout,
                concat=True,
            )
        )

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=attention_dropout,
                    concat=True,
                )
            )

        # Output layer (single head)
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                hidden_channels,
                heads=1,
                dropout=attention_dropout,
                concat=False,
            )
        )

        # Batch normalization
        self.bns = nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            else:
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        attention_weights = []

        for i, conv in enumerate(self.convs[:-1]):
            if return_attention:
                x, (edge_idx, alpha) = conv(
                    x, edge_index, return_attention_weights=True
                )
                attention_weights.append((edge_idx, alpha))
            else:
                x = conv(x, edge_index)

            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        if return_attention:
            x, (edge_idx, alpha) = self.convs[-1](
                x, edge_index, return_attention_weights=True
            )
            attention_weights.append((edge_idx, alpha))
        else:
            x = self.convs[-1](x, edge_index)

        out = self.classifier(x)

        if return_attention:
            return out, attention_weights
        return out

    def get_embeddings(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)

        x = self.convs[-1](x, edge_index)
        return x

    def get_attention_weights(self, x, edge_index):
        """Extract attention weights (useful for debugging predictions)."""
        _, attention_weights = self.forward(x, edge_index, return_attention=True)
        return attention_weights


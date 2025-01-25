"""GIN (Graph Isomorphism Network) for fraud detection.

Supposedly as powerful as the WL test for distinguishing graphs.
Uses MLPs inside the message passing which makes it more expressive
than simple mean/max aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class FraudGIN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, train_eps=True):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GIN layers with MLPs
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp, train_eps=train_eps))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Middle layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)

    def get_embeddings(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)

        return x


class GINWithJK(nn.Module):
    """GIN + Jumping Knowledge — concatenates representations from all layers.
    Helps when graph has varying depths/diameters."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, dropout=0.3, jk_mode="cat"):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.jk_mode = jk_mode

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # JK layer
        if jk_mode == "cat":
            jk_channels = hidden_channels * (num_layers + 1)
        else:
            jk_channels = hidden_channels

        if jk_mode == "lstm":
            self.jk_lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(jk_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_proj(x)
        layer_outputs = [x]

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        # Aggregate with JK
        if self.jk_mode == "cat":
            x = torch.cat(layer_outputs, dim=1)
        elif self.jk_mode == "max":
            x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
        elif self.jk_mode == "lstm":
            stacked = torch.stack(layer_outputs, dim=1)
            _, (h, _) = self.jk_lstm(stacked)
            x = h.squeeze(0)

        return self.classifier(x)

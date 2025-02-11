"""GraphSAGE model for fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGraphSAGE(nn.Module):
    """GraphSAGE for node-level fraud detection."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.3,
        aggregator="mean",
    ):
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


class EdgeFraudGraphSAGE(nn.Module):
    """Edge-level fraud detection — predicts per-transaction."""

    def __init__(
        self, in_channels, hidden_channels, edge_channels=0, num_layers=3, dropout=0.3
    ):
        super().__init__()

        self.node_encoder = FraudGraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Edge classifier
        edge_input_dim = 2 * hidden_channels + edge_channels
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2),
        )

    def forward(self, x, edge_index, edge_attr=None):
        # Get node embeddings
        node_emb = self.node_encoder.get_embeddings(x, edge_index)

        # Get source and target embeddings for each edge
        src_emb = node_emb[edge_index[0]]
        dst_emb = node_emb[edge_index[1]]

        # Concatenate edge features
        if edge_attr is not None:
            edge_repr = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
        else:
            edge_repr = torch.cat([src_emb, dst_emb], dim=1)

        # Classify edges
        return self.edge_classifier(edge_repr)

    def predict_proba(self, x, edge_index, edge_attr=None):
        logits = self.forward(x, edge_index, edge_attr)
        return F.softmax(logits, dim=1)[:, 1]


class NeighborSampler:
    """Mini-batch neighbor sampler for scalable training."""

    def __init__(self, edge_index, sizes, batch_size=512):
        self.edge_index = edge_index
        self.sizes = sizes
        self.batch_size = batch_size

    def __call__(self, node_idx: torch.Tensor):
        """Sample subgraph for given node indices."""
        from torch_geometric.utils import k_hop_subgraph

        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx,
            num_hops=len(self.sizes),
            edge_index=self.edge_index,
            relabel_nodes=True,
        )

        return subset, sub_edge_index, mapping

"""Heterogeneous GNN — handles multiple node types (user, merchant, device).

This was the trickiest part to get working. The edge type matching between
different node types kept breaking the forward pass. See the "wip" commit
in git history for the pain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroFraudGNN(nn.Module):
    """Heterogeneous GNN with per-type message passing."""

    def __init__(
        self,
        node_types,
        edge_types,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.3,
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers
        self.dropout = dropout

        # Node type embeddings (for nodes without features)
        self.node_embeddings = nn.ModuleDict()
        for node_type in node_types:
            self.node_embeddings[node_type] = nn.Embedding(10000, hidden_channels)

        # Linear projections for node features
        self.node_projections = nn.ModuleDict()
        for node_type in node_types:
            # Will be set dynamically based on input
            self.node_projections[node_type] = None

        # Heterogeneous convolutions
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(
                    hidden_channels, hidden_channels, aggr="mean"
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Batch normalization per node type
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            bn_dict = nn.ModuleDict()
            for node_type in node_types:
                bn_dict[node_type] = nn.BatchNorm1d(hidden_channels)
            self.bns.append(bn_dict)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def _project_features(self, x_dict, hidden_channels):
        """Project different node types to same hidden dim."""
        out = {}
        for node_type, x in x_dict.items():
            if x.dim() == 1:
                # Index-based: use embedding
                out[node_type] = self.node_embeddings[node_type](x)
            else:
                # Feature-based: use linear projection
                if self.node_projections[node_type] is None:
                    self.node_projections[node_type] = nn.Linear(
                        x.size(1), hidden_channels
                    ).to(x.device)
                out[node_type] = self.node_projections[node_type](x)

        return out

    def forward(self, x_dict, edge_index_dict):
        # Project features
        hidden_channels = self.convs[0].convs[self.edge_types[0]].in_channels
        x_dict = self._project_features(x_dict, hidden_channels)

        # Message passing
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            # Apply batch norm and activation
            for node_type in x_dict:
                x_dict[node_type] = self.bns[i][node_type](x_dict[node_type])
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(
                    x_dict[node_type], p=self.dropout, training=self.training
                )

        # Classification
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = self.classifier(x)

        return out_dict

    def get_embeddings(self, x_dict, edge_index_dict):
        hidden_channels = self.convs[0].convs[self.edge_types[0]].in_channels
        x_dict = self._project_features(x_dict, hidden_channels)

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            for node_type in x_dict:
                x_dict[node_type] = self.bns[i][node_type](x_dict[node_type])
                x_dict[node_type] = F.relu(x_dict[node_type])

        return x_dict


class HeteroEdgeFraudGNN(nn.Module):
    """Edge-level hetero GNN — classifies transaction edges."""

    def __init__(
        self,
        node_types,
        edge_types,
        hidden_channels,
        edge_channels=0,
        num_layers=3,
        dropout=0.3,
        target_edge_type=("user", "transacts", "merchant"),
    ):
        super().__init__()

        self.target_edge_type = target_edge_type

        self.node_encoder = HeteroFraudGNN(
            node_types=node_types,
            edge_types=edge_types,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Edge classifier
        src_type, _, dst_type = target_edge_type
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

    def forward(self, x_dict, edge_index_dict, edge_attr=None):
        # Get node embeddings
        emb_dict = self.node_encoder.get_embeddings(x_dict, edge_index_dict)

        # Get target edge type
        src_type, _, dst_type = self.target_edge_type
        edge_index = edge_index_dict[self.target_edge_type]

        # Get source and destination embeddings
        src_emb = emb_dict[src_type][edge_index[0]]
        dst_emb = emb_dict[dst_type][edge_index[1]]

        # Concatenate with edge features
        if edge_attr is not None:
            edge_repr = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
        else:
            edge_repr = torch.cat([src_emb, dst_emb], dim=1)

        return self.edge_classifier(edge_repr)

    def predict_proba(self, x_dict, edge_index_dict, edge_attr=None):
        logits = self.forward(x_dict, edge_index_dict, edge_attr)
        return F.softmax(logits, dim=1)[:, 1]


def create_hetero_model_from_data(data, hidden_channels=64, num_layers=3, dropout=0.3):
    """Helper to build a HeteroEdgeFraudGNN from a HeteroData object."""
    node_types = list(data.node_types)
    edge_types = list(data.edge_types)

    # Find transaction edge type
    target_edge_type = None
    for et in edge_types:
        if "transact" in et[1].lower():
            target_edge_type = et
            break

    if target_edge_type is None:
        target_edge_type = edge_types[0]

    # Get edge feature dimension
    edge_channels = 0
    if hasattr(data[target_edge_type], "edge_attr"):
        edge_attr = data[target_edge_type].edge_attr
        if edge_attr is not None:
            edge_channels = edge_attr.size(1)

    return HeteroEdgeFraudGNN(
        node_types=node_types,
        edge_types=edge_types,
        hidden_channels=hidden_channels,
        edge_channels=edge_channels,
        num_layers=num_layers,
        dropout=dropout,
        target_edge_type=target_edge_type,
    )

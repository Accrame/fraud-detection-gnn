"""GNNExplainer wrapper for interpreting fraud predictions."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


class FraudExplainer:
    """Wraps GNNExplainer to explain why a node/edge was flagged as fraud."""

    def __init__(self, model, data, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.data = data.to(self.device)

    def explain_node(self, node_idx, num_hops=2, epochs=100):
        """Use GNNExplainer to figure out why this node was predicted as fraud."""
        from torch_geometric.explain import Explainer, GNNExplainer

        # Create explainer
        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="log_probs",
            ),
        )

        # Get explanation
        explanation = explainer(
            self.data.x,
            self.data.edge_index,
            index=node_idx,
        )

        # Process explanation
        edge_mask = explanation.edge_mask.cpu().numpy()
        node_mask = (
            explanation.node_mask.cpu().numpy()
            if explanation.node_mask is not None
            else None
        )

        # Get k-hop subgraph
        from torch_geometric.utils import k_hop_subgraph

        subset, sub_edge_index, mapping, edge_mask_subset = k_hop_subgraph(
            node_idx,
            num_hops,
            self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes,
        )

        return {
            "node_idx": node_idx,
            "edge_mask": edge_mask,
            "node_mask": node_mask,
            "subgraph_nodes": subset.cpu().numpy(),
            "subgraph_edges": sub_edge_index.cpu().numpy(),
            "prediction": self._get_prediction(node_idx),
        }

    def explain_edge(self, edge_idx, num_hops=2):
        """Explain why a specific transaction was flagged."""
        # Get source and target nodes
        src = self.data.edge_index[0, edge_idx].item()
        dst = self.data.edge_index[1, edge_idx].item()

        # Get subgraph around the edge
        from torch_geometric.utils import k_hop_subgraph

        # Get neighborhood of both endpoints
        subset_src, _, _, _ = k_hop_subgraph(
            src, num_hops, self.data.edge_index, num_nodes=self.data.num_nodes
        )
        subset_dst, _, _, _ = k_hop_subgraph(
            dst, num_hops, self.data.edge_index, num_nodes=self.data.num_nodes
        )

        # Union of neighborhoods
        subset = torch.unique(torch.cat([subset_src, subset_dst]))

        # Compute feature importance using gradient
        feature_importance = self._compute_feature_importance(edge_idx)

        # Get neighboring edges
        edge_mask = torch.isin(self.data.edge_index[0], subset) & torch.isin(
            self.data.edge_index[1], subset
        )
        neighbor_edges = self.data.edge_index[:, edge_mask]

        return {
            "edge_idx": edge_idx,
            "src_node": src,
            "dst_node": dst,
            "subgraph_nodes": subset.cpu().numpy(),
            "neighbor_edges": neighbor_edges.cpu().numpy(),
            "feature_importance": feature_importance,
            "prediction": self._get_edge_prediction(edge_idx),
        }

    def _compute_feature_importance(self, edge_idx):
        """Gradient-based feature importance (not really integrated gradients, just vanilla)."""
        self.model.zero_grad()

        x = self.data.x.clone().requires_grad_(True)

        # Forward pass
        out = self.model(x, self.data.edge_index, getattr(self.data, "edge_attr", None))

        # Get prediction for target edge
        target_out = out[edge_idx, 1]  # Fraud probability

        # Backward
        target_out.backward()

        # Get gradients
        gradients = x.grad.abs().mean(dim=0).cpu().numpy()

        # Create feature importance dict
        feature_names = [
            "tx_count",
            "avg_amount",
            "std_amount",
            "max_amount",
            "unique_merchants",
            "merchant_tx_count",
            "merchant_avg_amount",
            "unique_users",
        ]

        importance = {}
        for i, name in enumerate(feature_names[: len(gradients)]):
            importance[name] = float(gradients[i])

        return importance

    def _get_prediction(self, node_idx: int) -> dict:
        """Get prediction for a node."""
        with torch.no_grad():
            out = self.model(
                self.data.x,
                self.data.edge_index,
                getattr(self.data, "edge_attr", None),
            )
            probs = F.softmax(out[node_idx], dim=0)

        return {
            "class": probs.argmax().item(),
            "fraud_prob": probs[1].item(),
            "confidence": probs.max().item(),
        }

    def _get_edge_prediction(self, edge_idx: int) -> dict:
        """Get prediction for an edge."""
        with torch.no_grad():
            out = self.model(
                self.data.x,
                self.data.edge_index,
                getattr(self.data, "edge_attr", None),
            )
            probs = F.softmax(out[edge_idx], dim=0)

        return {
            "class": probs.argmax().item(),
            "fraud_prob": probs[1].item(),
            "confidence": probs.max().item(),
        }

    def visualize(self, explanation, save_path=None, figsize=(12, 8)):
        """Plot the explanation subgraph + feature importance."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Subgraph visualization
        ax1 = axes[0]

        G = nx.Graph()
        edges = explanation.get("subgraph_edges", explanation.get("neighbor_edges"))

        if edges is not None:
            for i in range(edges.shape[1]):
                G.add_edge(edges[0, i], edges[1, i])

        pos = nx.spring_layout(G, seed=42)

        # Color nodes
        target_node = explanation.get("node_idx", explanation.get("src_node"))
        node_colors = []
        for node in G.nodes():
            if node == target_node:
                node_colors.append("red")
            elif node == explanation.get("dst_node"):
                node_colors.append("orange")
            else:
                node_colors.append("lightblue")

        nx.draw(
            G,
            pos,
            ax=ax1,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            font_size=8,
        )

        ax1.set_title("Subgraph Structure")

        # Feature importance
        ax2 = axes[1]

        feature_importance = explanation.get("feature_importance", {})
        if feature_importance:
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())

            # Sort by importance
            sorted_idx = np.argsort(importances)[::-1]
            features = [features[i] for i in sorted_idx]
            importances = [importances[i] for i in sorted_idx]

            ax2.barh(features, importances, color="steelblue")
            ax2.set_xlabel("Importance")
            ax2.set_title("Feature Importance")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def get_top_fraud_explanations(self, top_k=10):
        """Explain the top-k most suspicious predictions."""
        # TODO: cache explanations for frequently queried nodes
        with torch.no_grad():
            out = self.model(
                self.data.x,
                self.data.edge_index,
                getattr(self.data, "edge_attr", None),
            )
            probs = F.softmax(out, dim=1)[:, 1]

        # Get top-k fraud predictions
        top_indices = probs.topk(top_k).indices.cpu().numpy()

        explanations = []
        for idx in top_indices:
            exp = self.explain_edge(int(idx))
            explanations.append(exp)

        return explanations

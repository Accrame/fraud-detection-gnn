"""Inference module for fraud predictions."""

import time

import numpy as np
import torch


class FraudPredictor:
    """Loads a trained GNN and makes fraud predictions on new transactions."""

    def __init__(self, model_path, graph_builder=None, device="auto", threshold=0.5):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model(model_path)
        self.graph_builder = graph_builder
        self.threshold = threshold

        self.cached_graph = None
        self.inference_times = []

    def _load_model(self, model_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        from ..models.graphsage import EdgeFraudGraphSAGE

        model = EdgeFraudGraphSAGE(
            in_channels=checkpoint.get("in_channels", 8),
            hidden_channels=checkpoint.get("hidden_channels", 64),
            edge_channels=checkpoint.get("edge_channels", 7),
            num_layers=checkpoint.get("num_layers", 3),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def predict(self, transaction, return_details=False):
        """Predict fraud probability for a single transaction."""
        start_time = time.time()

        if self.cached_graph is None:
            raise ValueError("No graph cached. Call update_graph first.")

        with torch.no_grad():
            user_idx = self.graph_builder.user_mapping.get(transaction["user_id"])
            merchant_idx = self.graph_builder.merchant_mapping.get(
                transaction["merchant_id"]
            )

            if user_idx is None or merchant_idx is None:
                # new user/merchant — no graph context, default to 0.5
                fraud_prob = 0.5
            else:
                probs = self.model.predict_proba(
                    self.cached_graph.x.to(self.device),
                    self.cached_graph.edge_index.to(self.device),
                    (
                        self.cached_graph.edge_attr.to(self.device)
                        if hasattr(self.cached_graph, "edge_attr")
                        else None
                    ),
                )

                edge_mask = (self.cached_graph.edge_index[0] == user_idx) & (
                    self.cached_graph.edge_index[1] == merchant_idx
                )
                if edge_mask.any():
                    fraud_prob = probs[edge_mask][0].item()
                else:
                    fraud_prob = 0.5

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        risk_level = self._get_risk_level(fraud_prob)

        result = {
            "fraud_prob": fraud_prob,
            "risk_level": risk_level,
            "is_fraud": fraud_prob >= self.threshold,
            "inference_time_ms": inference_time * 1000,
        }

        if return_details:
            result["transaction"] = transaction
            result["threshold"] = self.threshold

        return result

    def predict_batch(self, transactions):
        """Predict fraud for a list of transactions."""
        return [self.predict(tx) for tx in transactions]

    def update_graph(self, graph):
        """Update the cached graph used for inference."""
        self.cached_graph = graph.to(self.device)

    def _get_risk_level(self, prob):
        if prob >= 0.9:
            return "critical"
        elif prob >= 0.7:
            return "high"
        elif prob >= 0.5:
            return "medium"
        elif prob >= 0.3:
            return "low"
        return "minimal"

    def get_statistics(self):
        """Basic inference latency stats."""
        if not self.inference_times:
            return {"avg_latency_ms": 0, "p50_latency_ms": 0, "p99_latency_ms": 0}

        times = np.array(self.inference_times) * 1000
        return {
            "avg_latency_ms": np.mean(times),
            "p50_latency_ms": np.percentile(times, 50),
            "p95_latency_ms": np.percentile(times, 95),
            "p99_latency_ms": np.percentile(times, 99),
            "total_predictions": len(self.inference_times),
        }

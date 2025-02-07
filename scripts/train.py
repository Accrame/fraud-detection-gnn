"""
Training script for fraud detection GNN.

Usage:
    python scripts/train.py --model graphsage --epochs 100
    python scripts/train.py --config configs/model_config.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_synthetic_fraud_data
from src.data.graph_builder import TransactionGraphBuilder
from src.models.gat import FraudGAT
from src.models.gin import FraudGIN
from src.models.graphsage import FraudGraphSAGE
from src.training.trainer import GNNTrainer, compute_class_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train fraud detection GNN")
    parser.add_argument(
        "--model",
        type=str,
        default="graphsage",
        choices=["graphsage", "gat", "gin"],
        help="Model architecture",
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--data", type=str, help="Path to transaction CSV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output", type=str, default="models/fraud_gnn.pt")
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_type, in_channels, hidden_channels, out_channels, num_layers, dropout):
    if model_type == "graphsage":
        return FraudGraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "gat":
        return FraudGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "gin":
        return FraudGIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        model_type = config.get("model", {}).get("type", args.model)
        hidden_dim = config.get("model", {}).get("hidden_channels", args.hidden_dim)
        num_layers = config.get("model", {}).get("num_layers", args.num_layers)
        dropout = config.get("model", {}).get("dropout", args.dropout)
        epochs = config.get("training", {}).get("epochs", args.epochs)
        lr = config.get("training", {}).get("learning_rate", args.lr)
    else:
        model_type = args.model
        hidden_dim = args.hidden_dim
        num_layers = args.num_layers
        dropout = args.dropout
        epochs = args.epochs
        lr = args.lr

    # Load or generate data
    if args.data:
        import pandas as pd
        transactions = pd.read_csv(args.data)
        print(f"Loaded {len(transactions)} transactions from {args.data}")
    else:
        print("No data provided, using synthetic data...")
        transactions = create_synthetic_fraud_data(
            num_users=1000,
            num_merchants=200,
            num_transactions=10000,
            fraud_rate=0.05,
        )

    fraud_rate = transactions["is_fraud"].mean()
    print(f"Fraud rate: {fraud_rate:.2%}")

    # Build graph
    builder = TransactionGraphBuilder()
    graph = builder.build_graph(transactions)
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

    # Create masks
    num_edges = graph.edge_index.shape[1] // 2
    train_mask, val_mask, test_mask = builder.get_train_test_masks(num_edges)
    graph.train_mask = torch.cat([train_mask, train_mask])
    graph.val_mask = torch.cat([val_mask, val_mask])
    graph.test_mask = torch.cat([test_mask, test_mask])

    # Build model
    in_channels = graph.x.shape[1]
    model = build_model(model_type, in_channels, hidden_dim, 2, num_layers, dropout)
    print(f"\nModel: {model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights for imbalanced data
    class_weights = compute_class_weights(graph.y)

    # Train
    trainer = GNNTrainer(
        model=model,
        data=graph,
        learning_rate=lr,
        class_weights=class_weights,
        device=args.device,
    )

    print(f"\nTraining for {epochs} epochs...")
    history = trainer.train(
        epochs=epochs,
        checkpoint_path=args.output,
    )

    # Evaluate
    print("\n--- Test Results ---")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()

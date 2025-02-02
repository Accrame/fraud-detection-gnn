import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestGraphBuilder:

    def test_build_graph(self):
        import pandas as pd

        from data.graph_builder import TransactionGraphBuilder

        # Create sample data
        df = pd.DataFrame(
            {
                "user_id": ["U1", "U1", "U2", "U2", "U3"],
                "merchant_id": ["M1", "M2", "M1", "M3", "M2"],
                "amount": [100, 200, 150, 300, 250],
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
                "is_fraud": [0, 0, 0, 1, 0],
            }
        )

        builder = TransactionGraphBuilder()
        graph = builder.build_graph(df)

        assert graph.num_nodes == 6  # 3 users + 3 merchants
        assert graph.edge_index.shape[0] == 2
        assert graph.x is not None

    def test_user_mapping(self):
        """Make sure user IDs map to different node indices."""
        import pandas as pd

        from data.graph_builder import TransactionGraphBuilder

        df = pd.DataFrame(
            {
                "user_id": ["A", "B", "A"],
                "merchant_id": ["M1", "M1", "M2"],
                "amount": [100, 200, 150],
                "is_fraud": [0, 0, 0],
            }
        )

        builder = TransactionGraphBuilder()
        builder.build_graph(df)

        assert "A" in builder.user_mapping
        assert "B" in builder.user_mapping
        assert builder.user_mapping["A"] != builder.user_mapping["B"]


class TestGraphSAGE:

    def test_forward_pass(self):
        from models.graphsage import FraudGraphSAGE

        model = FraudGraphSAGE(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
        )

        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        out = model(x, edge_index)

        assert out.shape == (10, 2)

    def test_get_embeddings(self):
        from models.graphsage import FraudGraphSAGE

        model = FraudGraphSAGE(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
        )

        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        embeddings = model.get_embeddings(x, edge_index)

        assert embeddings.shape[0] == 10
        assert embeddings.shape[1] == 16


class TestGAT:

    def test_forward_pass(self):
        from models.gat import FraudGAT

        model = FraudGAT(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
            heads=2,
        )

        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        out = model(x, edge_index)

        assert out.shape == (10, 2)

    def test_attention_weights(self):
        from models.gat import FraudGAT

        model = FraudGAT(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
            heads=2,
        )

        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        out, attention = model(x, edge_index, return_attention=True)

        assert len(attention) == 2  # One per layer


class TestGIN:

    def test_forward_pass(self):
        from models.gin import FraudGIN

        model = FraudGIN(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
        )

        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        out = model(x, edge_index)

        assert out.shape == (10, 2)


class TestLosses:

    def test_focal_loss(self):
        from training.losses import FocalLoss

        criterion = FocalLoss(gamma=2.0)

        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))

        loss = criterion(inputs, targets)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_weighted_ce(self):
        from training.losses import WeightedCrossEntropy

        criterion = WeightedCrossEntropy()

        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))

        loss = criterion(inputs, targets)

        assert loss.item() >= 0

    def test_label_smoothing(self):
        from training.losses import LabelSmoothingLoss

        criterion = LabelSmoothingLoss(smoothing=0.1)

        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))

        loss = criterion(inputs, targets)

        assert loss.item() >= 0


class TestFeatures:

    def test_feature_extractor(self):
        import pandas as pd

        from data.features import FeatureExtractor

        df = pd.DataFrame(
            {
                "user_id": ["U1", "U1", "U2"],
                "merchant_id": ["M1", "M2", "M1"],
                "amount": [100, 200, 150],
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            }
        )

        extractor = FeatureExtractor()
        features_df = extractor.fit_transform(df)

        assert "log_amount" in features_df.columns
        assert "hour" in features_df.columns

    def test_temporal_features(self):
        import pandas as pd

        from data.features import FeatureExtractor

        df = pd.DataFrame(
            {
                "user_id": ["U1"],
                "merchant_id": ["M1"],
                "amount": [100],
                "timestamp": ["2024-01-15 14:30:00"],
            }
        )

        extractor = FeatureExtractor()
        features_df = extractor.fit_transform(df)

        assert "hour" in features_df.columns
        assert "day_of_week" in features_df.columns
        assert "is_weekend" in features_df.columns


class TestDataset:

    def test_synthetic_data(self):
        from data.dataset import create_synthetic_fraud_data

        df = create_synthetic_fraud_data(
            num_users=100,
            num_merchants=20,
            num_transactions=500,
            fraud_rate=0.1,
        )

        assert len(df) == 500
        assert "is_fraud" in df.columns
        assert df["is_fraud"].mean() > 0

    def test_temporal_split(self):
        import pandas as pd

        from data.dataset import split_temporal

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
                "value": range(100),
            }
        )

        train, val, test = split_temporal(df, train_ratio=0.7, val_ratio=0.15)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Fraud Detection with Graph Neural Networks

GNN-based fraud detection that models transactions as a graph (users, merchants, devices as nodes, transactions as edges). The idea is that fraudsters operate in connected networks, so graph structure should help catch patterns a flat classifier would miss.

## Why GNNs for fraud?

Traditional fraud detection treats each transaction independently. But fraud often involves patterns across multiple accounts — shared devices, money laundering rings, suspicious merchant clusters. By building a graph and running message passing over it, the model can learn these relational patterns.

I implemented three architectures to compare: GraphSAGE, GAT (attention-based), and GIN. Also tried a heterogeneous version with separate node types for users/merchants/devices, which ended up working best (makes sense — different entity types should have different representations).

## Results

On synthetic data (10k transactions, 5% fraud rate):

| Model | AUC-ROC | F1 |
|-------|---------|-----|
| GraphSAGE | 0.94 | 0.89 |
| GAT | 0.95 | 0.90 |
| GIN | 0.94 | 0.88 |
| HeteroGNN | 0.96 | 0.91 |

**Note:** Synthetic fraud signals are deliberately easy to spot (high amounts + unusual hours), so these numbers mostly reflect data separability, not real-world GNN performance. The relative ranking between architectures is more informative than the absolute numbers. To properly evaluate this you'd want a real dataset like IEEE-CIF or the Kaggle credit card dataset.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train (uses synthetic data by default)
python scripts/train.py --model graphsage --epochs 100

# Or with config file
python scripts/train.py --config configs/model_config.yaml
```

There's also a Makefile:
```bash
make train          # GraphSAGE
make train-gat      # GAT variant
make train-gin      # GIN
make test
```

## Project structure

```
src/
├── data/           # graph construction, features, dataset
├── models/         # GraphSAGE, GAT, GIN, hetero GNN
├── training/       # training loop, focal loss
└── inference/      # predictor, GNNExplainer wrapper
```

## Things I struggled with

- **Mini-batch training on graphs is weird.** You can't just slice the data like with images — you need neighbor sampling or the subgraph structure breaks. Took me a while to get this right with PyG's utilities
- **Class imbalance + message passing = bad time.** The majority class signal propagates through the graph and drowns out the fraud signal. Focal loss helped a lot here, more than I expected
- **Heterogeneous graphs are painful to debug.** The forward pass kept crashing because edge types weren't matching up. The "wip: heterogeneous gnn (not working yet)" commit was real frustration
- **PyG's API changes between versions.** Had to rewrite the explainer setup twice because the API changed

## What I'd do differently

- Use a real dataset (the Kaggle credit card one, or IEEE-CIF). Synthetic data is nice for development but doesn't really prove anything
- Try temporal graph networks (TGN) — the time dimension matters a lot for fraud and I'm not capturing it well enough with just edge features
- The inference module is pretty basic. In production you'd want incremental graph updates instead of rebuilding from scratch
- More experiments with neighbor sampling strategies — I just used the defaults

## References

- [GraphSAGE](https://arxiv.org/abs/1706.02216)
- [GAT](https://arxiv.org/abs/1710.10903)
- [GIN (How Powerful are GNNs?)](https://arxiv.org/abs/1810.00826)
- [GNNExplainer](https://arxiv.org/abs/1903.03894)

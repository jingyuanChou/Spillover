# Spillover-Aware Simulation Analysis (Graph/Hypergraph SCI)

This repo contains code for **simulation-based evaluation of spillover effects** on networks, with models that can operate on a **plain graph** (pairwise edges)

The main entry point is `Spillover_model.py`.

---

## 1) Project structure

Core files (most important):
- **`Spillover_model.py`**: main training / evaluation script (ITE + spillover experiments, CLI args, training loop).
- **`Model.py`**: model definitions
- **`utils.py`**: utilities (projection hypergraph→graph, Wasserstein balancing distance, sparse conversions, etc.)
- **`data_simulation.py`**: simulation routines (generate outcomes / “true” counterfactual outcomes under neighbor flips).
- **`data_preprocessing.py`**: preprocessing for datasets (e.g., GoodReads / Microsoft contact style inputs).

Helper / analysis:
- `plots_change.py`: plotting utilities for results.
- `check_hypergraph.py`: sanity checks for hypergraph structure.
- `VA_counties.py`, `VA_label_match.py`: Virginia-specific data prep helpers (used when you run the VA scenario).

---

## 2) Environment

Recommended:
- Python **3.9+**
- PyTorch + (optional) CUDA

Key dependencies:
- `torch`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`
- `torch-geometric` (PyG), used by `GCNConv`, `GATConv`, and `HypergraphConv` in `Model.py`

### Install (example)
```bash
pip install numpy scipy pandas scikit-learn matplotlib
pip install torch
# Then install PyG following the official instructions for your CUDA / torch version:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-geometric
```
### 3) What the code is doing (logic)
Goal

Estimate potential outcomes and evaluate spillover when treatments of neighbors are flipped.

Representation / model

Both GraphSCI and HyperSCI share the same conceptual structure:

Feature encoder: phi_x = MLP(x)

Treatment-modulated features: phi_x_t = t * phi_x

Graph/Hypergraph encoder:

Graph: GCNConv(phi_x_t, edge_index)

Hypergraph: HypergraphConv(phi_x_t, hyperedge_index)

Concatenate: rep_post = [phi_x, rep_gnn_or_hgnn]

Two potential-outcome heads:

predict y0_pred and y1_pred

Factual prediction:

yf_pred = y1_pred if t=1 else y0_pred

Training loss (high level)

Inside Spillover_model.py, training optimizes a combination of:

Outcome prediction loss (factual outcome)

Balancing loss (Wasserstein distance between treated/control representations)

(Optional) extra graph-structure related terms depending on your configuration

Spillover evaluation (what makes it “spillover”)

For the spillover experiment:

load_data(...) returns:

pred_candidate: candidate “neighbor-flipped treatment vectors” (one per unit / fips)

real_after_flipping_neighbors: the corresponding true outcome after neighbor flips (from simulation / ground truth)

The code then calls:

model.predict(features, flipped_treatments, edge_index)

and compares:

true spillover = Y_true_after_flip - Y_factual

estimated spillover = Y_pred_after_flip - Y_factual

### 4) Data / paths
Default behavior

Spillover_model.py has a default --path pointing to a local .mat file (e.g., GoodReads simulation). You will likely need to change:

--path to your local dataset location, or

modify load_data() to load from your preferred files.

Important note

At the bottom of Spillover_model.py, the script overrides some CLI args in __main__:

sets args.graph_model = 'graph'

sets args.exp_name = 'spillover'

and forces projected graph settings.

So if you pass CLI args but the script still runs graph + spillover, it’s because of this override. If you want CLI to control it, remove/comment those lines in __main__.

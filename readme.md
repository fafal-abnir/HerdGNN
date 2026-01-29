# FistGNN

This repository contains the **official implementation** of the paper:

> **Mind the Deviations: Memory-Augmented Snapshot-Based GNNs  
> for Anomaly Detection in Evolving Graphs**  
> Vahid Shahrivari Joghan, Ramon Rico, Ioana Karnstedt-Hulpus, Yannis Velegrakis  
> Proceedings of the VLDB Endowment (PVLDB)

The paper studies **node-level and edge-level anomaly detection in discrete-time dynamic graphs** under extreme class imbalance and limited supervision.  
FistGNN introduces a **memory-augmented snapshot-based GNN** with a **deviation-aware objective**, enabling scalable and incremental learning without storing node histories.

---

## Overview

Real-world applications such as **financial fraud detection, anti-money laundering (AML), cybersecurity, and social platforms** generate large evolving graphs where anomalies are rare, labels are scarce, and models must be updated continuously.

FistGNN addresses these challenges by:
- Processing the graph as a **sequence of snapshots**
- Maintaining a **compact, graph-size-agnostic hierarchical memory**
- Learning anomaly scores via a **deviation-aware loss** that shapes the score distribution under extreme imbalance
- Supporting **live-update training**, where the model is incrementally updated without revisiting past snapshots

---

## Method Summary

FistGNN consists of four key components:

1. **Snapshot-Based GNN Backbone**  
   Each snapshot is processed independently using a standard GNN (e.g., GIN, GCN, GAT).

2. **Hierarchical Temporal Memory**  
   Instead of caching node embeddings over time, the model maintains a **fixed-size, per-layer memory** that evolves across snapshots and modulates GNN parameters.  
   The memory footprint is **independent of the number of nodes, edges, and snapshots**.

3. **Deviation-Aware Anomaly Scoring**  
   Risk scores of normal entities are modeled using **online Gaussian statistics**.  
   A deviation loss pushes labeled anomalies into the upper tail of the score distribution while keeping normal entities near the mean.

4. **Live-Update Training Protocol**  
   The model is incrementally fine-tuned snapshot by snapshot, carrying forward only:


---

## Repository Structure

All implementations follow a unified and modular design using **PyTorch Lightning**.

For simplicity and clarity, each method is organized around **three core files**:

1. **`main.py`**  
   Entry point for running experiments.  
   - Loads experiment parameters  
   - Initializes datasets and models  
   - Launches training and evaluation  

2. **`lightning_module.py`**  
   Implements the training, validation, and testing strategy.  
   Includes:
   - forward pass  
   - loss computation (classification + deviation loss)  
   - evaluation metrics  
   - optimizer and scheduler setup  

3. **`model.py`**  
   Contains the implementation of the GNN architecture.

---

## Dataset Processing

The `dataset/` directory contains all preprocessing and data-loading code.

Processing includes:
- Transforming raw event streams into **discrete-time graph snapshots**
- Preparing data for training with PyTorch Geometric in DGNN setup

This design allows consistent evaluation across multiple datasets and tasks.

---

## Implemented Models

The repository includes:

- **FistGNN** (proposed method)
- Temporal GNN baselines (ROLAND, WinGNN, HawkGNN) and their integration with deviation-loss

---

## Getting Started

### Prerequisites

- Python **3.10** or higher
- [Poetry](https://python-poetry.org/) for dependency management

---

### Installation

Clone the repository:

```bash
git clone -
cd FistGNN
```

Create and activate the virtual environment and install dependencies:
```bash
poetry shell
poetry install
```
### Running Experiments
There are examples of commands for running each method on different dataset for example for running our method on Elliptic dataset:
```bash
python3 herdnet_main.py  --epochs=200 --learning_rate=0.005 --alph=0.05 --blend_factor=0.9 --dropout=0.1 --gnn_type=GIN --dataset="EllipticPP"  --hidden_size=128 --memory_size=256 --num_windows=49 --force_reload_dataset --enable_memory;
```
note that `--force_reload_dataset` force to preprocess dataset again before running the experiment 
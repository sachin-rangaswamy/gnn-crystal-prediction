# Graph Neural Networks for Crystal Property Prediction

![GitHub](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.3-brightgreen)

This project uses **Graph Neural Networks (GNNs)** to predict material properties of crystalline structures. By representing crystals as graphs (atoms as nodes and bonds as edges), the model captures local and global atomic interactions to accurately predict properties such as bandgap and formation energy.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

Crystalline materials are naturally represented as graphs, where atoms are nodes and bonds are edges. This project leverages **Graph Neural Networks (GNNs)** to model these structures and predict material properties. The GNN incorporates **attention mechanisms** to weigh important atomic interactions and handles **3D periodic boundary conditions** to accurately represent crystal structures.

---

## Key Features

- **Graph Representation**: Crystals are represented as graphs, with atoms as nodes and bonds as edges.
- **Attention Mechanisms**: Uses Graph Attention Networks (GAT) to focus on important atomic interactions.
- **Periodic Boundary Conditions**: Handles 3D periodicity of crystals by considering bonds across unit cell boundaries.
- **Scalability**: Can be extended to large datasets of crystalline materials.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- pymatgen

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gnn-crystal-prediction.git
   cd gnn-crystal-prediction





## Acknowledgments

This project was made possible thanks to the following resources and tools:

- **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)**: A library for deep learning on graphs and irregular structures.
- **[Materials Project](https://materialsproject.org/)**: A database of crystal structures and material properties.
- **[pymatgen](https://pymatgen.org/)**: A Python library for materials analysis.
- **[NumPy](https://numpy.org/)**: For numerical computations in Python.
- **[Matplotlib](https://matplotlib.org/)**: For creating visualizations.
- **[scikit-learn](https://scikit-learn.org/)**: For machine learning tools and utilities.

Special thanks to the open-source community for their contributions and support!

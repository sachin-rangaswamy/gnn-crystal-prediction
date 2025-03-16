# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

# Step 1: Load Crystalline Data
# Example: Load CIF files from the Materials Project
def load_crystal_data(cif_path):
    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]
    return structure

# Step 2: Convert Crystals to Graphs
def crystal_to_graph(structure, radius=5.0):
    # Node features: Atomic number, atomic mass, electronegativity
    node_features = []
    for site in structure:
        atomic_number = site.specie.Z
        atomic_mass = site.specie.atomic_mass
        electronegativity = site.specie.X
        node_features.append([atomic_number, atomic_mass, electronegativity])
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge indices and features: Bonds based on distance
    edge_indices = []
    edge_features = []
    for i, site_i in enumerate(structure):
        for j, site_j in enumerate(structure):
            if i != j:
                distance = structure.get_distance(i, j)
                if distance <= radius:
                    edge_indices.append([i, j])
                    edge_features.append([distance])
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    # Create PyTorch Geometric Data object
    graph = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_features)
    return graph

# Step 3: Define a GNN Model with Attention
class CrystalGNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(CrystalGNN, self).__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Fully connected layer
        x = self.fc(x)
        return x

# Step 4: Prepare Dataset and DataLoader
class CrystalDataset(Dataset):
    def __init__(self, cif_files, targets):
        super(CrystalDataset, self).__init__()
        self.cif_files = cif_files
        self.targets = targets

    def len(self):
        return len(self.cif_files)

    def get(self, idx):
        structure = load_crystal_data(self.cif_files[idx])
        graph = crystal_to_graph(structure)
        graph.y = torch.tensor([self.targets[idx]], dtype=torch.float)
        return graph

# Example: Load CIF files and targets
cif_files = ["material1.cif", "material2.cif", "material3.cif"]  # Replace with actual CIF files
targets = [1.2, 0.8, 1.5]  # Example bandgap values
dataset = CrystalDataset(cif_files, targets)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 5: Train the GNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrystalGNN(node_dim=3, edge_dim=1, hidden_dim=64, output_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # Train for 10 epochs
    model.train()
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Step 6: Evaluate the Model
model.eval()
with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        print("Predicted:", out.cpu().numpy())
        print("Actual:", batch.y.cpu().numpy())

# Step 7: Visualize Attention Weights (Optional)
# Extract attention weights from the GATConv layers

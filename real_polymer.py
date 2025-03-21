import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

# Sample Polymer Dataset (Synthetic Data for Demonstration)
class PolymerDataset(Dataset):
    def __init__(self, num_samples=1000):
        np.random.seed(42)
        self.X = np.random.rand(num_samples, 15)  # 15 polymer structure features
        self.y = 3 * np.sin(self.X.sum(axis=1)) + np.random.normal(0, 0.1, num_samples)  # Target property

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Define the AI Model for Polymer Property Prediction
class PolymerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolymerNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Training and Evaluation Functions
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss/len(test_loader):.4f}")

# Data Preparation
dataset = PolymerDataset(num_samples=1000)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model Initialization
input_dim = 15
hidden_dim = 64
output_dim = 1

model = PolymerNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation
train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, test_loader, criterion)

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
from sklearn.model_selection import cross_validate
from sklearn.datasets import make_classification

# Define PyTorch model
class PyTorchNN(nn.Module):
    def __init__(self, num_features):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create a PyTorch model wrapper compatible with scikit-learn
class SklearnPyTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=10, batch_size=32, learning_rate=0.01, random_seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_seed(self, random_seed):
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

    def get_params(self, deep=True):
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'random_seed': self.random_seed
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        self.set_seed(self.random_seed)  # Ensure random seed is set each time fit is called
        self.model = PyTorchNN(num_features=X.shape[1])
        self.model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
        # Convert DataFrame to NumPy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
    
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    
        self.classes_ = np.unique(y)  # Set classes_ attribute
        self.model.train()
    
        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
    
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(dataloader)}")
    
        return self

    def predict_proba(self, X):
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy()
        return np.hstack((1 - outputs, outputs))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
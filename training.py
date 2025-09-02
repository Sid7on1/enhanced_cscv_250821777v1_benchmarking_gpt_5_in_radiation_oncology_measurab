import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    'model': 'LSTM',
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'input_dim': 10,
    'output_dim': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class AgentDataset(Dataset):
    def __init__(self, data: pd.DataFrame, labels: pd.Series):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        label = self.labels.iloc[idx]
        return {
            'input': torch.tensor(sample.values, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class AgentModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super(AgentModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(CONFIG['device'])
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(CONFIG['device'])
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class AgentTrainer:
    def __init__(self, model: AgentModel, device: str):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=CONFIG['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(CONFIG['epochs']):
            logger.info(f'Epoch {epoch+1} of {CONFIG["epochs"]}')
            start_time = time.time()
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                inputs, labels = batch['input'].to(self.device), batch['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            logger.info(f'Training loss: {total_loss / len(train_loader)}')
            self.model.eval()
            val_loss = 0
            predictions = []
            labels = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels_batch = batch['input'].to(self.device), batch['label'].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels_batch)
                    val_loss += loss.item()
                    predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    labels.extend(labels_batch.cpu().numpy())
            logger.info(f'Validation loss: {val_loss / len(val_loader)}')
            logger.info(f'Validation accuracy: {accuracy_score(labels, predictions)}')
            logger.info(f'Validation F1 score: {f1_score(labels, predictions, average="macro")}')
            logger.info(f'Validation AUC score: {roc_auc_score(labels, predictions)}')
            self.scheduler.step(accuracy_score(labels, predictions))
            end_time = time.time()
            logger.info(f'Time taken for epoch {epoch+1}: {end_time - start_time} seconds')

def main():
    # Load data
    data = pd.read_csv('data.csv')
    labels = pd.read_csv('labels.csv')

    # Split data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create dataset and data loaders
    train_dataset = AgentDataset(train_data, train_labels)
    val_dataset = AgentDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Create model and trainer
    model = AgentModel(CONFIG['input_dim'], CONFIG['output_dim'], CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout'])
    trainer = AgentTrainer(model, CONFIG['device'])

    # Train model
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
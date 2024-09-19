import torch
import torch.nn as nn
import torch.nn.functional as F

class AslGruModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=3, hidden_size=32):
        super(AslGruModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, X):
        X, _ = self.gru(X)
        X = F.relu(X)

        X = X.permute(0, 2, 1)
        X = self.pool(X)
        X = X.squeeze(-1)

        X = self.fc(X)

        return X
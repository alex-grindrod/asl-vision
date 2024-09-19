import torch
import torch.nn as nn
import torch.nn.functional as F
import math

FC = torch.ones((50, 50))

class TimeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
    

    def forward(self, X):
        #Expected input: N H W C (batch size, num nodes, num timesteps, num features)
        X_tens = X.permute(0, 3, 1, 2)
        X = self.conv1(X_tens)
        X += F.sigmoid(self.conv2(X_tens))
        X += self.conv3(X_tens)
        X = F.relu(X)
        return X.permute(0, 2, 3, 1)
    

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_channels, num_nodes, adj_matrix=FC):
        super(STGCNBlock, self).__init__()
        self.time1 = TimeConv(in_channels, out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.time2 = TimeConv(spatial_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.adj = torch.tensor(adj_matrix, dtype=torch.float32) if isinstance(adj_matrix, list) else adj_matrix
        self.reset_parameters()
    

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    
    def forward(self, X):
        X = self.time1(X)
        X = torch.einsum("ij, jklm->kilm", [self.adj, X.permute(1, 0, 2, 3)])
        X = torch.matmul(X, self.Theta1)
        X = F.relu(X)
        X = self.time2(X)
        return self.batch_norm(X)
    

class AslSTGCNModel(nn.Module):
    def __init__(self, num_classes, num_features, num_timesteps, num_nodes, adj_matrix=FC):
        super(AslSTGCNModel, self).__init__()
        self.adj = adj_matrix
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64, spatial_channels=16, num_nodes=num_nodes, adj_matrix=self.adj)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64, spatial_channels=16, num_nodes=num_nodes, adj_matrix=self.adj)
        self.time = TimeConv(in_channels=64, out_channels=64)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear((num_timesteps - 2 * 5) * 64, num_classes)
        self.num_classes = num_classes

    
    def forward(self, X):
        X = X.permute(0, 2, 1, 3)
        X = self.block1(X)
        X = self.block2(X)
        X = self.time(X)
        X = X.reshape((X.shape[0], X.shape[1], -1))
        X = X.permute(0, 2, 1)
        X = self.avg_pool(X).squeeze(-1)
        X = self.fc(X)
        
        return X
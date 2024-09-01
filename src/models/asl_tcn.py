import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation):
        super(TCNLayer, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )

        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = F.relu(x)
        return x

class AslTcnModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=3, hidden_size=64, kernel_size=3):
        super(AslTcnModel, self).__init__()
        layers = []
        dilation = 1
        for i in range(num_layers):
            layers.append(TCNLayer(input_size, hidden_size, kernel_size, dilation))
            input_size = hidden_size
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.mean(dim=2)
        return self.fc(x)
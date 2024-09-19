from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AslLstmModel(pl.LightningModule):
    def __init__(self, input_size, output_size, num_layers=3, hidden_size=64):
        super(AslLstmModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=self.num_layers)

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_features=hidden_size, out_features=64))
        self.fc_layers.append(nn.Linear(in_features=64, out_features=32))

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_out= nn.Linear(in_features=32, out_features=output_size)

        self.loss_fn = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=output_size, average='macro')
        self.validation_step_outputs = []

        self.count = 0
        print(f"Using: {device}")


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        x, _ = self.lstm(x, (h0, c0))
        x = F.relu(x)

        x = x.permute(0, 2, 1)
        x = self.max_pool(x).squeeze(-1)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)
        
        x = self.fc_out(x)
        x = F.softmax(x, dim=1)
        return x

# from typing import Any
# import pytorch_lightning as pl
# from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchmetrics

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class AslLstmModel(pl.LightningModule):
#     def __init__(self, input_size, output_size, num_layers=3, hidden_size=64):
#         super(AslLstmModel, self).__init__()
#         self.lstm_layers = nn.ModuleList()
#         self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True, num_layers=1))
#         self.lstm_layers.append(nn.LSTM(input_size=64, hidden_size=128, batch_first=True, num_layers=1))
#         self.lstm_layers.append(nn.LSTM(input_size=128, hidden_size=64, batch_first=True, num_layers=1))

#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         # self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=self.num_layers)

#         self.fc_layers = nn.ModuleList()
#         self.fc_layers.append(nn.Linear(in_features=hidden_size, out_features=64))
#         self.fc_layers.append(nn.Linear(in_features=64, out_features=32))

#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#         self.fc_out= nn.Linear(in_features=32, out_features=output_size)

#         self.loss_fn = nn.CrossEntropyLoss()
#         self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=output_size, average='macro')
#         self.validation_step_outputs = []

#         self.count = 0
#         print(f"Using: {device}")


#     def forward(self, x):
#         print(x.shape)
#         batch_size = x.size(0)

#         # h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
#         # c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

#         # x, _ = self.lstm_layers[0](x, (h0, c0))
#         # x = x[:, -1, :]
#         # x = F.relu(x)

#         # h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
#         # c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

#         # x, _ = self.lstm_layers[1](x, (h0, c0))
#         # x = x[:, -1, :]
#         # x = F.relu(x)

#         # h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
#         # c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)

#         # x, _ = self.lstm_layers[2](x, (h0, c0))
#         # x = x[:, -1, :]
#         # x = F.relu(x)
        

#         for lstm in self.lstm_layers:
#             h0 = torch.zeros(1, batch_size, lstm.hidden_size).to(x.device)
#             c0 = torch.zeros(1, batch_size, lstm.hidden_size).to(x.device)

#             x, _ = lstm(x, (h0, c0))
#             x = x[:, -1, :]
#             x = F.relu(x)

#         # x = x.permute(0, 2, 1)
#         # x = self.max_pool(x).squeeze(-1)

#         for fc_layer in self.fc_layers:
#             x = fc_layer(x)
#             x = F.relu(x)
        
#         x = self.fc_out(x)
#         x = F.softmax(x, dim=1)
#         return x

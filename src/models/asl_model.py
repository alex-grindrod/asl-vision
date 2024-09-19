from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from src.models.asl_lstm import AslLstmModel
from src.models.asl_tcn import AslTcnModel
from src.models.asl_gru import AslGruModel
from src.models.asl_st_gcn import AslSTGCNModel
from src.models.asl_st_gcn_orig import Asl_STGCN_18_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AslModel(pl.LightningModule):
    def __init__(self, model_type, input_size, output_size, learning_rate, model_params):
        super(AslModel, self).__init__()
        self.save_hyperparameters()

        self.input_size=input_size
        self.output_size=output_size
        self.lr = learning_rate

        if model_type == 'AslLstm':
            self.model = AslLstmModel(self.input_size, self.output_size, **model_params)
        elif model_type == 'AslTcn':
            self.model = AslTcnModel(self.input_size, self.output_size, **model_params)
        elif model_type == 'AslGru':
            self.model = AslGruModel(self.input_size, self.output_size, **model_params)
        elif model_type == 'AslSTGCN':
            self.model = AslSTGCNModel(self.output_size, **model_params)
        elif model_type == 'AslSTGCN18':
            self.model = Asl_STGCN_18_Model(self.input_size, self.output_size, **model_params)
        else:
            raise ValueError("No valid model type specified")
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=output_size, average='macro')
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=output_size, average='macro')



    def forward(self, X):
        X = torch.nan_to_num(X)
        # X = X.to(torch.float32)
        return self.model(X)
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)

        prediction = torch.argmax(y_pred, dim=1)
        print(f"Thing: {prediction} \n {y}")

        accuracy = torchmetrics.functional.accuracy(prediction, y, task='multiclass', num_classes=self.output_size, average='macro')
        print(f'train acc: {accuracy}')
        if self.logger:
            self.log('Training Loss', loss)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        print(y_pred.shape)
        print(y.shape)
        loss = self.loss_fn(y_pred, y)

        if self.logger:
            self.log('val_loss', loss)
        
        prediction = torch.argmax(y_pred, dim=1)
        accuracy = torchmetrics.functional.accuracy(prediction, y, task='multiclass', num_classes=self.output_size, average='macro')

        self.val_accuracy.update(prediction, y)

        print(f"Normal val acc: {accuracy}")
        return {"val_loss" : loss, "val_accuracy" : accuracy}


    def on_validation_epoch_end(self):
        val_accuracy = self.val_accuracy.compute()

        if self.logger:
            self.log('val_accuracy', val_accuracy)

        print(f"Ending val acc: {val_accuracy}")
        
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
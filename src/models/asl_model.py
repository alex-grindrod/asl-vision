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
        prediction = torch.argmax(y_pred, dim=1)
        print(f"Thing: {prediction} \n {y}")
        print(self.train_accuracy(prediction, y))
        return self.loss_fn(y_pred, y)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        loss = self.loss_fn(y_pred, y)

        if self.logger:
            self.log('val_loss', loss)
        
        prediction = torch.argmax(y_pred, dim=1)
        accuracy = self.val_accuracy(prediction, y)

        self.val_accuracy.update(y_pred, y)

        print(f"Normal val acc: {accuracy}")
        return {"val_loss" : loss, "val_accuracy" : accuracy}


    def on_validation_epoch_end(self):
        val_accuracy = self.val_accuracy.compute()

        if self.logger:
            self.log('val_accuracy', val_accuracy)

        print(f"Ending val acc: {val_accuracy}")
        
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
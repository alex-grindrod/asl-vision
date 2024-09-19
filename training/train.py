import sys
sys.path.append(".")

import pytorch_lightning as pl
import wandb
import torch
from argparse import ArgumentParser
from pathlib import Path
from src.asl_data.asl_datamodule import ASLDataModule
from src.models.asl_model import AslModel
from training.callbacks import GradientMonitoringCallback
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from src.annotation.mp_annotate import setup_adj_mat
from src.utilities import ASLConfig

MODEL_SAVE_PATH = Path.cwd() / 'models'


def train(args : ASLConfig):
    num_classes = args.subset
    model_type = args.model_type

    if args.wandb:
        wandb.init(project="ASL_VISION", entity='agrindro')
        wandb_logger = pl.loggers.WandbLogger(name='Trial 1', project="ASL_VISION", entity='agrindro')
    else:
        print("Skipping wandb")
        wandb_logger=None

    data_module = ASLDataModule(
        dataset_path=args.processed_dir,
        subset=num_classes,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        model_type=model_type
    )
    data_module_shape = data_module.setup()

    model_params = {
        'AslLstm': {
            'num_layers' : 1,
            'hidden_size' : 32,
        },
        'AslTcn' : {
            'num_layers' : 5,
            'hidden_size' : 128,
            'kernel_size' : 3
        },
        'AslGru' : {
            'num_layers' : 1,
            'hidden_size' : 128
        },
        'AslSTGCN' : {
            'num_features' : 2,
            'num_timesteps' : 50,
            'num_nodes' : 50,
            'adj_matrix' : torch.ones((50, 50))
        },
        'AslSTGCN18' : {
            'adj_matrix' : torch.ones((1, 50, 50), requires_grad=False),
            'edge_importance_weighting' : args.edge_importance_weighting,
            'data_bn' : args.data_bn,
            'dropout' : args.dropout,
            'residual' : args.residual
        } 
    }
    input_size = 2 if model_type == 'AslSTGCN18' else data_module_shape[-1]
    model = AslModel(model_type=model_type, input_size=input_size, output_size=num_classes, learning_rate=args.learning_rate, model_params=model_params[model_type])

    dirpath = MODEL_SAVE_PATH / model_type
    if not dirpath.exists():
        dirpath.mkdir(parents=True)

    callbacks = [
        # GradientMonitoringCallback(),
        ModelCheckpoint(
            dirpath=dirpath,
            filename='{val_accuracy}_best',
            save_top_k=1,
            verbose=True,
            monitor='val_accuracy',
            mode='max',
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger
    )

    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

if __name__ == '__main__':
    config = ASLConfig()
    parser = ArgumentParser()

    parser.add_argument('-e', '--max_epochs', type=int, default=config.max_epochs, help='Number of epochs to train for.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=config.learning_rate, help='The learning rate for training model.')
    parser.add_argument('-dw', '--wandb', action='store_false', help='Enable wandb (default: True)')

    print("Training")
    args = parser.parse_args()
    config = ASLConfig(**args.__dict__)
    train(config)

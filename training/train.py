import sys
sys.path.append(".")

import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser, Namespace
from pathlib import Path
from src.asl_datamodule import ASLDataModule
from src.models.asl_model import AslModel
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)


DEFAULT_DATASET = Path("data/processed_unsplit_30")
MODEL_TYPE = "asl_lstm"


def train(args):
    num_classes = 10
    if args.wandb:
        wandb.init(project="ASL_VISION", entity='agrindro')
        wandb_logger = pl.loggers.WandbLogger(name='Trial 1', project="ASL_VISION", entity='agrindro')
    else:
        print("Skipping wandb")
        wandb_logger=None

    data_module = ASLDataModule(
        dataset_path=DEFAULT_DATASET,
        subset=num_classes,
        batch_size=128,
        val_split=0.2,
        test_split=0.1,
        num_workers=4
    )
    data_module.setup()

    model_params = {
        'AslLstm': {
            'num_layers' : 5,
            'hidden_size' : 128,
        },
        'AslTcn' : {
            'num_layers' : 5,
            'hidden_size' : 128,
            'kernel_size' : 3
        }
    }
    

    model = AslModel(model_type='AslLstm', input_size=744, output_size=num_classes, learning_rate=args.learning_rate, model_params=model_params['AslLstm'])

    callbacks = [
        ModelCheckpoint(
            dirpath=Path.cwd() / 'models' / 'asl_lstm',
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
        accelerator='auto',
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger
    )

    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--max_epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='The learning rate for training model.')
    parser.add_argument('-w', '--wandb', action='store_false', help='Enable wandb (default: True)')

    print("Training")
    args = parser.parse_args()
    train(args)

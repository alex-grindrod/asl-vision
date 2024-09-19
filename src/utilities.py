import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

ROOT = Path.cwd()


@dataclass
class ASLConfig:
    model_type : str = 'AslTcn' # Possible types: AslLstm | AslTcn | AslGru | AslSTGCN | AslSTGCN18

    #Data Related Config
    processed_dir : Path = ROOT / 'data' / 'processed_WLASL3_50'
    raw_dir : Path = ROOT / 'WLASL2000'
    subset : int = 3
    val_split : float = 0.2
    test_split : float = 0.1
    num_workers : int = 4

    #Data Preprocessing Related Config
    frame_cap : int = 50
    resolution : Tuple[int, int] = (256, 256)
    interpolation : int = cv2.INTER_LINEAR
    num_features : int = 2
    variant_split : bool = True

    #General Training Related Config
    max_epochs : int = 50
    learning_rate : int = 0.001
    wandb : bool = False
    accelerator: str = 'auto'
    batch_size : int = 64

    #AslSTGCN18 train config
    edge_importance_weighting : bool = True
    data_bn : bool = True
    dropout : float = 0.2
    residual : bool = True

    


import sys
sys.path.append(".")

import pytorch_lightning as pl
from pathlib import Path
from src.asl_data.asl_dataset import ASLDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

DEFAULT_DATASET = Path("data/processed_unsplit_30")

class ASLDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: Path, subset: int = 100, batch_size: int = 32, val_split: int = 0.2, test_split: int = 0.1, num_workers: int = 4, model_type: str = 'AslLstm'):
        super().__init__()

        self.dataset_path = dataset_path
        if not self.dataset_path.exists() or not self.dataset_path.is_dir():
            raise Exception("Invalid path: not found or is not directory")
        
        self.subset = subset
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.model_type = model_type
        self.tranform = transforms.Compose([
            transforms.ToTensor()
        ])

    
    def setup(self, stage=None):
        full_asldataset = ASLDataset(dataset_path=self.dataset_path, subset=self.subset, model_type=self.model_type)
        full_asllabels = [label for _, label in full_asldataset]

        total_size = len(full_asldataset)
        indices = list(range(total_size))

        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_split,
            stratify=full_asllabels,
            shuffle=True
        )

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=self.val_split / (1 - self.test_split),
            stratify=[full_asllabels[i] for i in train_val_indices],
            shuffle=True
        )
        
        self.train_dataset = Subset(full_asldataset, train_indices)
        self.val_dataset = Subset(full_asldataset, val_indices)
        self.test_dataset = Subset(full_asldataset, test_indices)

        return full_asldataset.shape()
    
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        

if __name__ == "__main__":
    data_module = ASLDataModule(
        dataset_path=DEFAULT_DATASET,
        subset=100,
        batch_size=32,
        val_split=0.2,
        test_split=0.1,
        num_workers=4
    )
    data_module.setup()

    print("Train split:", sum(len(batch[0]) for batch in data_module.train_dataloader()))
    print("Val split:", sum(len(batch[0]) for batch in data_module.val_dataloader()))
    print("Test split:", sum(len(batch[0]) for batch in data_module.test_dataloader()))
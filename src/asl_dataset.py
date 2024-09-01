import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset


class ASLDataset(Dataset):
    def __init__(self, dataset_path: Path, subset: int = 100):
        """
          data : mediapipe keypoint annotations (Tensor)
        labels : corresponding American english word (String)
        """
        self.data = []
        self.labels = []

        with open(str(dataset_path / "_word_ids.json"), 'r') as file:
            word_priority_dict = json.load(file)

        for word in tqdm(word_priority_dict, desc="Extracting data"):
            word_path = dataset_path / word
            for annotation in word_path.iterdir():
                if annotation.suffix == ".npy":
                    np_data = np.load(str(annotation))
                    self.data.append(torch.tensor(np_data, dtype=torch.float32))
                    self.labels.append(word_priority_dict[word])

            if word_priority_dict[word] >= subset - 1:
                break
        
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
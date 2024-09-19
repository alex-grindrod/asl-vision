import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset


class ASLDataset(Dataset):
    def __init__(self, dataset_path: Path, data_format: str ='mediapipe', model_type: str ='AslLstm', subset: int = 100):
        """
          data : mediapipe keypoint annotations (Tensor)
        labels : corresponding American english word (String)
        """
        self.data = []
        self.labels = []

        with open(str(dataset_path / "_word_ids.json"), 'r') as file:
            word_priority_dict = json.load(file)

        if data_format == 'mediapipe':
            for word in tqdm(word_priority_dict, desc="Extracting data"):
                word_path = dataset_path / word
                for annotation in word_path.iterdir():
                    if annotation.suffix == ".npy":
                        np_data = np.load(str(annotation))
                        self.data.append(torch.tensor(np_data, dtype=torch.float32))
                        self.labels.append(word_priority_dict[word])

                if word_priority_dict[word] >= subset:
                    break
        elif data_format == 'raw video':
            pass
        
        # mini = 2
        # maxi = 0
        # off = 0
        # for ind, sequence in tqdm(enumerate(self.data), desc="Finding range"):
        #     for keypoints in sequence:
        #         for value in keypoints.numpy():
        #             if value < 0 or value > 1:
        #                 off += 1
        #             mini = min(mini, value)
        #             maxi = max(maxi, value)
        # print(mini, maxi)
        # print(off)
        # exit()


        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        if model_type == 'AslSTGCN18':  # self.data : (Batch size, frames, nodes, features) --> (Batch size, features, frames, nodes, num graphs per vid)
            self.data = self.data.permute(0, 3, 1, 2)
            self.data = self.data.unsqueeze(-1) # If adding more graphs later, need to change (Not that smart yet)

        # print(self.data.shape) 
        # print(self.labels.shape)
        # exit()


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
    
    def shape(self):
        return self.data.shape
import pandas as pd
from torch.utils.data import DataLoader

class Dataset():
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "id": self.data["id"][idx],
            "context": self.data["context"][idx],
            "claim": self.data["claim"][idx]
        }
        return sample

def getDataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
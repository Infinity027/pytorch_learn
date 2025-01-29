import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = torch.tensor(row[:-1], dtype=torch.float32)
        label = torch.tensor(row[-1], dtype=torch.float32) 
        return features, label

    def __len__(self):
        return len(self.dataframe)


if __name__=="__main__":
    df = pd.read_csv("out.csv", float_precision='high')
    print(df.head())
    data = CustomDataset(dataframe=df)
    dataloader = DataLoader(data, batch_size=5)
    for sample in dataloader:
        print(sample)
        break
import os
import ast

import torch
import gzip
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from config import Config


def get_folds(df) -> pd.DataFrame:
    study_ids = df["study_id"].unique()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    df["fold"] = -1

    for fold, (train_idx, test_idx) in enumerate(kf.split(study_ids)):
        fold_studies = study_ids[test_idx]
        df.loc[df["study_id"].isin(fold_studies), "fold"] = fold

    return df

def convert_to_array(string_data):
    formatted_data = string_data.replace(' ', ',')
    list_data = ast.literal_eval(formatted_data)
    return np.array(list_data, dtype=int)

class StudyPerVertebraLevelDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):

        self.df = dataframe
        self.df["label_as_array"] = [e for e in self.df["label"].str.replace(" ", "").str.split(",") if e != ""]
        print(self.df["label_as_array"].head())
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with gzip.open(row["path"], "rb") as f:
            volume = np.load(f, allow_pickle=True)
        label = row["label_as_array"]

        volume = torch.FloatTensor(volume)
        volume = self.transform(volume)

        return volume, torch.tensor(label, dtype=torch.long)
    
    


if __name__ == "__main__":
    df = get_folds(pd.read_csv(Config.data_basepath + "processed_studies/preprocessed_train.csv"))
    ds = StudyPerVertebraLevelDataset(df)
    for i in range(3):
        volume, label = ds[i]
        print(volume.shape, label)
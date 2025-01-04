import pandas as pd

dataset = pd.read_csv("dataset/races_filled.csv")

dataset["label"] = (dataset["position"] >= 20).astype(int)

TR = dataset[dataset["race_year"] >= 2022]

TS = dataset[dataset["race_year"] < 2022]

TR.to_csv("dataset_classification/TrainDataset.csv")
TS.to_csv("dataset_classification/TestDataset.csv")
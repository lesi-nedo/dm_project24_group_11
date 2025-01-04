import pandas as pd

dataset = pd.read_csv("dataset/races_filled.csv")

dataset["label"] = (dataset["position"] < 20).astype(float)
dataset["points"] = dataset["points"].astype(float)
dataset["startlist_quality"] = dataset["startlist_quality"].astype(float)

features_to_mantain = set(['points', 'uci_points', 'length', 'climb_total', 'profile',
                           'startlist_quality', 'cyclist_age', 'delta', 'label'])

TR = dataset[dataset["race_year"] < 2022]
TS = dataset[dataset["race_year"] >= 2022]

for i in set(dataset.keys()) - features_to_mantain:
    
    del TR[i]
    del TS[i]

TR.to_csv("dataset_classification/TrainDataset.csv", index = False)
TS.to_csv("dataset_classification/TestDataset.csv", index = False)
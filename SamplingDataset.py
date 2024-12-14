import pandas as pd

for i in range(1, 11):

    Perc = i/10

    train_data = pd.read_csv('DatasetClassification/TrainDataset.csv')
    test_data = pd.read_csv('DatasetClassification/TestDataset.csv')

    print(len(train_data))
    print(len(test_data))

    train_data = train_data.sample(frac = Perc)
    test_data = train_data.sample(frac = Perc)

    print("-"*100)

    print(len(train_data))
    print(len(test_data))

    train_data.to_csv(f"DatasetClassification/TrainDataset{int(Perc*100)}%.csv", index=False)
    test_data.to_csv(f"DatasetClassification/TestDataset{int(Perc*100)}%.csv", index=False)
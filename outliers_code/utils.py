import pandas as pd

from fastai.tabular.all import TabularPandas, Categorify, Normalize



def encode_tabular_data(dataset, cat_features, cont_features):
    """
    Encode a dataset using TabularPandas with categorical and continuous features.
    Automatically converts categorical features to category dtype.
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        The input dataset to encode
    cat_features : list
        List of categorical feature names - these will be converted to category dtype
    cont_features : list
        List of continuous feature names
        
    Returns:
    --------
    pandas.DataFrame
        Encoded dataset with processed categorical and continuous features
        
    Examples:
    --------
    >>> encoded_df = encode_tabular_data(
    ...     dataset=df,
    ...     cat_features=['nationality', 'gender', 'education'],
    ...     cont_features=['age', 'salary']
    ... )
    """
    # Create a copy with only required features
    df_for_encoding = dataset[[*cat_features, *cont_features]].copy()
    
    # Convert all categorical features to category dtype
    for col in cat_features:
        try:
            df_for_encoding[col] = df_for_encoding[col].astype('category')
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert {col} to category type. Error: {e}")
    
    # Create TabularPandas object
    dls = TabularPandas(
        df_for_encoding,
        procs=[Categorify, Normalize],
        cat_names=cat_features,
        cont_names=cont_features,
        y_names=None,
        splits=None
    )
    
    # Get encoded dataframe
    df_encoded = dls.train.xs
    
    return df_encoded


def get_races_outliers(MergedDataset: pd.DataFrame, all_outliers: pd.DataFrame) -> pd.DataFrame:
    al_races_outliers = all_outliers.groupby('name_race').size().reset_index(name='count').sort_values(by=["name_race"], ascending=False)
    all_part_out_race = MergedDataset[MergedDataset['name_race'].isin(all_outliers['name_race'])].groupby(["name_race"]).size().reset_index(name='count').sort_values(by=["name_race"], ascending=False)


    races_outliers = al_races_outliers[(al_races_outliers['name_race'] == all_part_out_race['name_race']) & ((al_races_outliers['count'] / all_part_out_race['count']) > 0.5)]

    return races_outliers 


def get_cyclists_outliers(MergedDataset: pd.DataFrame, all_outliers: pd.DataFrame) -> pd.DataFrame:
    al_cyclists_outliers = all_outliers.groupby('name_cyclist').size().reset_index(name='count').sort_values(by=["name_cyclist"], ascending=False)
    all_part_out_cyclist = MergedDataset[MergedDataset['name_cyclist'].isin(all_outliers['name_cyclist'])].groupby(["name_cyclist"]).size().reset_index(name='count').sort_values(by=["name_cyclist"], ascending=False)


    cyclists_outliers = al_cyclists_outliers[(al_cyclists_outliers['name_cyclist'] == all_part_out_cyclist['name_cyclist']) & ((al_cyclists_outliers['count'] / all_part_out_cyclist['count']) > 0.5)]

    return cyclists_outliers


def get_cleaned_dataset() -> pd.DataFrame:
    # Load outliers
    outliers_files = ["dataset/outliers_lof.csv", "dataset/outliers_iso_for.csv", "dataset/outliers_oc_svm.csv"]
    all_outliers = pd.concat([pd.read_csv(file) for file in outliers_files]).drop_duplicates(subset=['name_race', 'name_cyclist'])

    # Load datasets
    DatasetCyclists = pd.read_csv("dataset/cyclists_filled.csv")
    DatasetRaces = pd.read_csv("dataset/races_filled.csv")

    # Merge datasets
    MergedDataset = DatasetRaces.merge(DatasetCyclists, left_on='cyclist', right_on='_url', how='inner')
    MergedDataset.rename(columns={"_url_y": "name_cyclist", "_url_x": "name_race"}, inplace=True)
    MergedDataset.drop(columns=['cyclist', 'name_x', 'name_y'], inplace=True)

    # Get outliers
    races_outliers = get_races_outliers(MergedDataset, all_outliers)
    cyclist_outliers = get_cyclists_outliers(MergedDataset, all_outliers)

    # Remove outliers
    DatasetCyclists = DatasetCyclists[~DatasetCyclists['_url'].isin(cyclist_outliers['name_cyclist'].unique())]
    DatasetRaces = DatasetRaces[~DatasetRaces["_url"].isin(races_outliers['name_race'].unique())]

    # Merge datasets again
    MergedDataset = DatasetRaces.merge(DatasetCyclists, left_on='cyclist', right_on='_url', how='inner')
    MergedDataset.rename(columns={"_url_y": "name_cyclist", "_url_x": "name_race"}, inplace=True)
    MergedDataset.drop(columns=['cyclist', 'name_x', 'name_y'], inplace=True)

    return MergedDataset
   
    

__all__ = [
    'encode_tabular_data'
    'get_races_outliers',
    'get_cyclists_outliers',
    'get_cleaned_dataset'
    ]
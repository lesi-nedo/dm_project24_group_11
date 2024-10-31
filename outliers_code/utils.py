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

__all__ = ['encode_tabular_data']
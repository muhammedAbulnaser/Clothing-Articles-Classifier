import os


def clean_unexisting_files(df, DATASET_PATH):
    """
    Clean unexisting files from the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        DATASET_PATH (str): Path to the dataset directory.

    Returns:
        pd.DataFrame: Cleaned DataFrame with existing files.

    """
    problems = []
    for idx, line in df.iterrows():
        if not os.path.exists(os.path.join(DATASET_PATH, 'images', str(line.id)+'.jpg')):
            print(idx)
            problems.append(idx)
    df.drop(df.index[problems], inplace=True)

    return df


def filter_dataframe(df):
    """
    Filter the DataFrame based on a category count threshold.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        pd.DataFrame: Filtered DataFrame with categories that have counts above the threshold.

    """
    # Compute the count of each category in the 'articleType' column
    category_counts = df['articleType'].value_counts()

    # Filter the DataFrame to include only categories with counts above 1000
    filtered_df = df[df['articleType'].isin(category_counts[category_counts > 1000].index)]

    return filtered_df


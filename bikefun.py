# helper functions

def clean_col_names(df):

    """
    Remove upper-case letters and : from data with OSM tags
    Special characters like ':' can for example break with pd.query function

    Arguments:
        df (df/gdf): dataframe/geodataframe with OSM tag data

    Returns:
        df (df/gdf): the same dataframe with updated column names
    """

    df.columns = df.columns.str.lower()

    df_cols = df.columns.to_list()

    new_cols = [c.replace(":", "_") for c in df_cols]

    df.columns = new_cols

    return df
def density(df, column):
    return df[column].count() / len(df)

def intersect_columns(l, df):
    """
    Make sure that elements of the list l are columns of the DataFrame df.
    Args:
        - l  : a list supposed to contain columns;
        - df : a DataFrame where l's elements are supposed to come from.

    Returns:
         - a list: intersection between l and df.columns.
    """
    return [e for e in l if e in df.columns]

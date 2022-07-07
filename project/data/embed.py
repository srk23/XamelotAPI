# Transform the data into a more ML-friendly shape.

import re
from project.data.encode import OneHotEncoder


def embed_data(df, duration, descriptor, separator='#'):
    """
    Prepare the data to be injected into a machine learning model.
    Notably, it:

    - One Hot Encode categorical columns;
    - Standardize numerical data;
    - Convert types to `float32`;

    Args:
        - df                    : an input DataFrame
        - duration : columns known to have heterogeneous typing
        - descriptor      : values that are generally used to mark unknown values
        - separator     : values that are used to mark unknown values, per column
    Returns:
        A machine learning ready to use version of the DataFrame provided as input.
    """
    # One Hot Encoding
    ohe = OneHotEncoder(descriptor, separator=separator)

    df = ohe.encode(df)

    # Standardize numerical data
    columns_to_standardize = [col for col in df.columns if
                              descriptor.get_entry(re.split(ohe.separator, col)[0]).is_numerical]
    columns_to_standardize = list(set(columns_to_standardize) - {duration})

    df[columns_to_standardize] = df[columns_to_standardize].apply(lambda s: (s - s.mean()) / s.std())

    # Convert types to float32
    df = df.astype('float32')

    return df

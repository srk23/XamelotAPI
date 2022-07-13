# Provide a set of tools to clean raw datasets.

from project.data.parameters import HETEROGENEOUS_COLUMNS, \
                                    GENERIC_UNKNOWNS,      \
                                    SPECIFIC_UNKNOWNS,     \
                                    LIMITS,                \
                                    REFERENCES
from project.misc.dataframes import build_empty_mask, intersect_columns


###################
#      CLEAN      #
###################

# Set column names to lower case
def set_columns_to_lower_case(df):
    return df.rename(str.lower, axis='columns')

# Change type to handle non-float NaN
def change_int64(df):
    dtype = {
            column: 'Int64'
            for column in df.columns
            if df.dtypes[column] == 'int64'
        }
    return df.astype(dtype)

# Ensure type uniformity
def ensure_type_uniformity(df, heterogeneous_columns):
    columns_to_retype = intersect_columns(heterogeneous_columns, df)
    df[columns_to_retype] = df[columns_to_retype].applymap(str)
    return df

# Change 'unknown values' to NaN
def correct_unknown_values(df, mask, generic_unknowns, specific_unknowns):
    mask |= (df.isin(generic_unknowns))
    for column, nan_values in specific_unknowns.items():
        if column in df.columns:
            mask[column] |= df[column].isin(nan_values)
    return df, mask

# Remove abnormal values (see 'limits' dict)
def remove_abnormal_values(df, mask, limits):
    for columns, minmax_values in limits:
        min_value, max_value = minmax_values
        columns_to_crop = intersect_columns(columns, df)
        mask[columns_to_crop] |= df[columns_to_crop] < min_value
        mask[columns_to_crop] |= df[columns_to_crop] > max_value
    return df, mask

# Use category names instead of codes
def use_category_names(df, references):
    for columns, reference in references:
        def _remap_(value):
            if value in reference.keys():
                return reference[value]
            else:
                return value

        columns_to_remap = intersect_columns(columns, df)  # list(columns & set(clean_df.columns.to_list()))
        df[columns_to_remap] = df[columns_to_remap].applymap(_remap_)
    return df

def clean_data(
    df, 
    heterogeneous_columns = HETEROGENEOUS_COLUMNS,
    generic_unknowns      = GENERIC_UNKNOWNS,
    specific_unknowns     = SPECIFIC_UNKNOWNS,
    limits                = LIMITS,
    references            = REFERENCES
):
    """
    Perform a basic cleaning of the data.
    Notably, it:

    - Set column names to lower case;
    - Change type to handle non-float NaN;
    - Ensure type uniformity;
    - Change 'unknown values' to NaN;
    - Remove abnormal values;
    - Use category names instead of codes.

    Note that this cleaning step is designed as little intrusive and destructive as possible towards the data.

    Args:
        - df                    : an input DataFrame
        - heterogeneous_columns : columns known to have heterogeneous typing
        - generic_unknowns      : values that are generally used to mark unknown values
        - specific_unknowns     : values that are used to mark unknown values, per column
        - limits                : valid ranges for numerical values
        - references            : mapping betzeen categorical codes and humanly readable labels
    Returns:
        A cleaner version of the DataFrame provided as input.
    """

    clean_df = df.copy()
        
    # Set column names to lower case    
    clean_df = set_columns_to_lower_case(clean_df)
                
    # Change type to handle non-float NaN
    clean_df = change_int64(clean_df)
    
    # Ensure type uniformity
    clean_df = ensure_type_uniformity(clean_df, heterogeneous_columns)
    
    # Change 'unknown values' to NaN
    mask = build_empty_mask(clean_df)
    clean_df, mask = correct_unknown_values(clean_df, mask, generic_unknowns, specific_unknowns)
    
    # Remove abnormal values (see 'limits' dict)
    clean_df, mask = remove_abnormal_values(clean_df, mask, limits)

    clean_df = clean_df.mask(mask)
            
    # Use category names instead of codes
    clean_df = use_category_names(clean_df, references)

    return clean_df

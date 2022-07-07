# Provide a set of tools to clean raw datasets.

from project.data.variables  import HETEROGENEOUS_COLUMNS, \
                                    GENERIC_UNKNOWNS,      \
                                    SPECIFIC_UNKNOWNS,     \
                                    LIMITS,                \
                                    REFERENCES
from project.misc.dataframes import intersect_columns

# Clean dataframes
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
    clean_df = clean_df.rename(str.lower, axis='columns')
                
    # Change type to handle non-float NaN
    dtype = {
            column: 'Int64'
            for column in clean_df.columns 
            if clean_df.dtypes[column] == 'int64'
        }
    clean_df = clean_df.astype(dtype)
    
    # Ensure type uniformity    
    columns_to_retype = intersect_columns(heterogeneous_columns, clean_df)  # list(heterogeneous_columns \
    # & set(clean_df.columns.to_list()))
    clean_df[columns_to_retype] = clean_df[columns_to_retype].applymap(str)
    
    # Change 'unknown values' to NaN    
    mask = (clean_df.isin(generic_unknowns))
    for column, nan_values in specific_unknowns.items():
        if column in clean_df.columns:
            mask[column] |= clean_df[column].isin(nan_values)
    
    # Remove abnormal values (see 'maxs' dict)  
    for columns, minmax_values in limits:
        min_value, max_value = minmax_values
        columns_to_crop = intersect_columns(columns, clean_df)
        mask[columns_to_crop] |= clean_df[columns_to_crop] < min_value
        mask[columns_to_crop] |= clean_df[columns_to_crop] > max_value

    clean_df = clean_df.mask(mask)
            
    # Use category names instead of codes
    for columns, reference in references:
        def _remap_(value):
            if value in reference.keys():
                return reference[value]
            else:
                return value
        
        columns_to_remap = intersect_columns(columns, clean_df)  # list(columns & set(clean_df.columns.to_list()))
        clean_df[columns_to_remap] = clean_df[columns_to_remap].applymap(_remap_)

    return clean_df

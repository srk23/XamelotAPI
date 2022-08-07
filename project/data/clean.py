# Provide a set of tools to clean raw datasets.
import numpy  as np
import pandas as pd

from project.data.build      import build_egfr, build_max, build_min, build_easy_trend, build_binary_code
from project.data.parameters import get_biolevel_columns

from project.misc.clinical   import compute_bmi
from project.misc.dataframes import build_empty_mask, intersect_columns, get_constant_columns


############################
#      CLEANING STEPS      #
############################


# Set column names to lower case
def set_columns_to_lower_case(df, cpm=None):
    _ = cpm
    return df.rename(str.lower, axis='columns')


# Change type to handle non-float NaN
def change_int64(df, cpm=None):
    _ = cpm
    dtype = {
            column: 'Int64'
            for column in df.columns
            if df.dtypes[column] == 'int64'
        }
    return df.astype(dtype)


# Ensure type uniformity
def ensure_type_uniformity(df, cpm):
    columns_to_retype = intersect_columns(cpm.heterogeneous_columns, df)
    df[columns_to_retype] = df[columns_to_retype].applymap(str)
    return df


# Change 'unknown values' to NaN
def correct_unknown_values(df, cpm):
    mask  = build_empty_mask(df)
    mask |= (df.isin(cpm.generic_unknowns))
    for column, nan_values in cpm.specific_unknowns.items():
        if column in df.columns:
            mask[column] |= df[column].isin(nan_values)
    return df.mask(mask)


# Remove abnormal values (see 'limits' dict)
def remove_abnormal_values(df, cpm):
    mask = build_empty_mask(df)
    for columns, minmax_values in cpm.limits:
        min_value, max_value = minmax_values
        columns_to_crop = intersect_columns(columns, df)
        mask[columns_to_crop] |= df[columns_to_crop] < min_value
        mask[columns_to_crop] |= df[columns_to_crop] > max_value
    return df.mask(mask)


# Use category names instead of codes
def use_category_names(df, cpm):
    for columns, reference in cpm.references:
        def _remap_(value):
            if value in reference.keys():
                return reference[value]
            else:
                return value

        columns_to_remap = intersect_columns(columns, df)  # list(columns & set(clean_df.columns.to_list()))
        df[columns_to_remap] = df[columns_to_remap].applymap(_remap_)
    return df


# Impute BMI
def impute_bmi(df, cpm):
    min_threshold, max_threshold = cpm.bmi_limits

    for prefix in ('r', 'd'):
        bmi    = prefix + "bmi"
        weight = prefix + "weight"
        height = prefix + "height"

        # Imputation
        condition = df[bmi].isna()
        other     = compute_bmi(df[weight], df[height])
        df[bmi]   = df[bmi].mask(condition, other)

        # Additional cleaning
        condition  = (min_threshold < df[bmi]) & (df[bmi] < max_threshold)
        df[weight] = df[weight].where(condition, pd.NA)
        df[height] = df[height].where(condition, pd.NA)
        df[bmi]    = df[bmi].where(condition, pd.NA)
    return df


# Transform dialysis related columns
def transform_dialysis_columns(df, cpm):
    is_positive_or_negative = {
        "days_on_dial_tx": lambda x: 0 if x > 0 else 1,
        "dial_at_reg"    : lambda x: 1 if x == "Not on dialysis" else 0,
        "dial_at_tx_type": lambda x: 1 if x == "Not on dialysis" else 0,
        "dial_at_tx"     : lambda x: cpm.descriptor.get_entry("dial_at_tx").binary_keys[x]
    }
    df["dial_code"] = build_binary_code(
        df,
        list(is_positive_or_negative.keys()),
        is_positive_or_negative
    )

    df["dial_type"] = df["dial_at_tx_type"]
    df["dial_days"] = df["days_on_dial_tx"]

    df.loc[df['dial_code'].isin([3, 27, 30, 39, 41]), "dial_days"] = 0
    df.loc[df['dial_code'].isin([29, 32])           , "dial_days"] = np.nan

    df.loc[df['dial_code'].isin([3, 6, 8, 30, 33, 35, 44, 62]), "dial_type"] = df["dial_at_reg"]
    df.loc[df['dial_code'] == 27, "dial_type"] = "Not on dialysis"

    return df.drop(columns=['dial_at_reg', 'dial_at_tx', 'dial_at_tx_type', 'days_on_dial_tx', 'dial_code'])


# Recompute eGFR
def recompute_egfr(df, cpm=None):
    _ = cpm
    creatinines = get_biolevel_columns('creatinine', df)
    egfrs       = get_biolevel_columns('degfr'     , df)
    creat_egfr  = zip(creatinines, egfrs)

    for creatinine, egfr in creat_egfr:
        s = build_egfr(df, creatinine)
        df[egfr] = df[egfr].where(s.isna(), s)
    return df


# Get trends, minimum, and maximum
def impute_biolevels(df, cpm=None):
    def _columns_(key_, df_, temporal_columns_only=False):
        columns_ = get_biolevel_columns(key_, df_, temporal_columns_only=temporal_columns_only)
        return intersect_columns(columns_, df_)

    _ = cpm

    columns_to_drop = list()
    for key in ('alt', 'ast', 'amylase', 'creatinine', 'degfr'):
        # Get trends
        columns = _columns_(key, df, temporal_columns_only=True)
        df[key + "_trend"] = build_easy_trend(df, columns)

        # Get min / max levels
        columns = _columns_(key, df, temporal_columns_only=False)
        df[key + "_min"] = build_min(df, columns)
        df[key + "_max"] = build_max(df, columns)

        columns_to_drop += columns
    return df.drop(columns=columns_to_drop)


# Add an "Unknown" category
def add_unknown_category(df, cpm):
    df[cpm.columns_with_unknowns] = df[cpm.columns_with_unknowns].mask(
        df[cpm.columns_with_unknowns].isna(),
        cpm.unknown
    )

    return df


# Remove irrelevant_columns
def remove_irrelevant_categories(df, cpm):
    columns = intersect_columns(cpm.irrelevant_categories.keys(), df)

    for column in columns:
        df = df.drop(df[df[column].isin(cpm.irrelevant_categories[column])].index)

    return df


# Remove irrelevant columns
def remove_irrelevant_columns(df, cpm):
    def _is_irrelevant_(column):
        return cpm.descriptor.get_entry(column).tags not in {"feature", "target"} \
                or column in cpm.irrelevant_columns

    columns_to_drop = intersect_columns(
        [column for column in df.columns if _is_irrelevant_(column)],
        df
    )
    return df.drop(columns=columns_to_drop)


# Remove constant columns
def remove_constant_columns(df, cpm=None):
    _ = cpm
    return df.drop(columns=get_constant_columns(df))


# Reorder columns
def reorder_columns(df, cpm):
    cols1 = [col for col in df.columns if cpm.descriptor.get_entry(col).tags != "target"]
    cols2 = [col for col in df.columns if cpm.descriptor.get_entry(col).tags == "target"]

    return df[cols1 + cols2]


###################
#      CLEAN      #
###################


CLEANING_STEPS = (
    set_columns_to_lower_case,
    change_int64,
    ensure_type_uniformity,
    correct_unknown_values,
    remove_abnormal_values,
    use_category_names,
    impute_bmi,
    transform_dialysis_columns,
    recompute_egfr,
    impute_biolevels,
    add_unknown_category,
    remove_irrelevant_categories,
    remove_constant_columns,
    reorder_columns
)


def clean_data(
    df,
    cleaning_steps              = CLEANING_STEPS,
    cleaning_parameters_manager = None
):
    """

    """

    clean_df = df.copy()

    for cleaning_step in cleaning_steps:
        clean_df = cleaning_step(clean_df, cleaning_parameters_manager)

    return clean_df

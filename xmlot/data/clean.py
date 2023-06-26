# Provide a set of tools to clean raw datasets.

import numpy  as np
import pandas as pd

from xmlot.data.build      import build_egfr,           \
                                  build_max,            \
                                  build_min,            \
                                  build_easy_trend,     \
                                  build_base_and_max,   \
                                  build_binary_prd,     \
                                  build_binary_graft_no,\
                                  build_binary_code,    \
                                  build_pcens,          \
                                  build_psurv,          \
                                  build_tcens,          \
                                  build_tsurv,          \
                                  build_mcens,          \
                                  build_msurv
from xmlot.data.dataframes import build_empty_mask, \
                                  intersect_columns,\
                                  get_constant_columns

from xmlot.misc.clinical   import compute_bmi


#############################
#    AUXILIARY FUNCTIONS    #
#############################


def get_biolevel_columns(biolevel, df, temporal_columns_only=False):
    """
        Provides all the original columns corresponding to a given biological level (AST, creatinine, etc.).

        Args:
            - biolevel              : a biolevel name
            - df                    : a DataFrame
            - temporal_columns_only : allow to only consider columns for which there is a temporal ordering (up to 63)

        Returns: the list of all the columns corresponding to the investigated biolevel that are present in df.
    """
    idx     = [11, 12, 31, 32, 61, 62, 63, 71, 72, 73, 74, 75, 81, 82, 83, 84, 85]
    columns = [biolevel + '_' + str(i) for i in idx]
    if temporal_columns_only:
        columns = columns[:7]
    else:
        if biolevel == "creatinine":
            additional_column = ['dret_creat']
        elif biolevel == "degfr":
            additional_column = ['degfr']
        else:
            additional_column = list()
        columns = additional_column + columns
    return intersect_columns(columns, df)


############################
#      CLEANING STEPS      #
############################


def set_columns_to_lower_case(df, **_):
    """
    Set column names to lower case.

    Args:
        - df : a DataFrame.

    Returns: the updated DataFrame.
    """
    return df.rename(str.lower, axis='columns')


def change_int64(df, **_):
    """
    Change type to handle non-float NaN.

    Args:
        - df : a DataFrame.

    Returns: the updated DataFrame.
    """
    dtype = {
            column: 'Int64'
            for column in df.columns
            if df.dtypes[column] == 'int64'
        }
    return df.astype(dtype)


def ensure_type_uniformity(df, heterogeneous_columns, **_):
    """
    Ensure type uniformity.

    Args:
        - df                    : a DataFrame;
        - heterogeneous_columns : a list of columns that contains values of various types.

    Returns: the updated DataFrame.
    """
    columns_to_retype = intersect_columns(heterogeneous_columns, df)
    df[columns_to_retype] = df[columns_to_retype].applymap(str)
    return df


def correct_unknown_values(df, generic_unknowns, specific_unknowns, **_):
    """
    Change 'unknown values' to NaN.

    Args:
        - df                : a DataFrame;
        - generic_unknowns  : a list of values that refer to an unknown value across the whole dataset;
        - specific_unknowns : a dictionary that specifies the list of values referring to an unkown value per column.

    Returns: the updated DataFrame.
    """
    mask  = build_empty_mask(df)
    mask |= (df.isin(generic_unknowns))
    for column, nan_values in specific_unknowns.items():
        if column in df.columns:
            mask[column] |= df[column].isin(nan_values)
    return df.mask(mask)


def remove_abnormal_values(df, limits, **_):
    """
    Remove abnormal values (see 'limits' dict).

    Args:
        - df     : a DataFrame;
        - limits : a dictionary that provides for the appropriate columns a pair of minimum and maximum values.

    Returns: the updated DataFrame.
    """
    mask = build_empty_mask(df)
    for columns, minmax_values in limits:
        min_value, max_value = minmax_values
        columns_to_crop = intersect_columns(columns, df)
        mask[columns_to_crop] |= df[columns_to_crop] <= min_value
        mask[columns_to_crop] |= df[columns_to_crop] >= max_value
    return df.mask(mask)


def use_category_names(df, references, **_):
    """
    Use category names instead of codes.

    Args:
        - df         : a DataFrame;
        - references : a dictionary that provides a textual correspondence of some 'encoded' categories.

    Returns: the updated DataFrame.
    """
    for columns, reference in references:
        def _remap_(value):
            if value in reference.keys():
                return reference[value]
            else:
                return value

        columns_to_remap = intersect_columns(columns, df)  # list(columns & set(clean_df.columns.to_list()))
        df[columns_to_remap] = df[columns_to_remap].applymap(_remap_)
    return df


def impute_bmi(df, bmi_limits, **_):
    """
    Impute BMI.

    Args:
        - df         : a DataFrame;
        - bmi_limits : provides a pair of minimum and maximum values regarding BMI.

    Returns: the updated DataFrame.
    """
    min_threshold, max_threshold = bmi_limits

    for prefix in ('r', 'd', 'rreg'):
        bmi    = prefix + "bmi"
        weight = prefix + "weight"
        height = prefix + "height"

        if bmi in df.columns:
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


def transform_dialysis_columns(df, descriptor, **_):
    """
    Transform dialysis related columns.

    Args:
        - df         : a DataFrame;
        - descriptor : a Descriptor.

    Returns: the updated DataFrame.
    """
    is_positive_or_negative = {
        "days_on_dial_tx": lambda x: 0 if x > 0 else 1,
        "dial_at_reg"    : lambda x: 1 if x == "Not on dialysis" else 0,
        "dial_at_tx_type": lambda x: 1 if x == "Not on dialysis" else 0,
        "dial_at_tx"     : lambda x: descriptor.get_entry("dial_at_tx").categorical_keys[x]
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

    # Minor adjustments for the "Not on dialysis" entries
    # Unknown dial_days are set to zero when dial_type is "Not on dialysis".
    df["dial_days"] = df["dial_days"].mask(
        (df["dial_type"] == "Not on dialysis")
        & df["dial_days"].isna(),
        other=0
    )

    # dial_days and dial_type are set to NaN when they have contradictory values ("Not on dialysis" vs >0 dial days).
    df[["dial_days", "dial_type"]] = df[["dial_days", "dial_type"]].mask(
        (df["dial_type"] == "Not on dialysis")
        & df["dial_days"] > 0,
        other=np.nan
    )

    return df.drop(columns=['dial_at_reg', 'dial_at_tx', 'dial_at_tx_type', 'days_on_dial_tx', 'dial_code'])


def recompute_egfr(df, **_):
    """
    Recompute eGFR.

    Args:
        - df : a DataFrame.

    Returns: the updated DataFrame.
    """
    creatinines = get_biolevel_columns('creatinine', df)
    egfrs       = get_biolevel_columns('degfr'     , df)
    creat_egfr  = zip(creatinines, egfrs)

    for creatinine, egfr in creat_egfr:
        s = build_egfr(df, creatinine)
        df[egfr] = df[egfr].where(s.isna(), s)
    return df


def impute_biolevels(df, **_):
    """
    Get trends, minimum, and maximum for biolevels.

    Args:
        - df : a DataFrame.

    Returns: the updated DataFrame.
    """
    # def _columns_(key_, df_, temporal_columns_only=False):
    #     columns_ = get_biolevel_columns(key_, df_, temporal_columns_only=temporal_columns_only)
    #     return intersect_columns(columns_, df_)
    #
    # columns_to_drop = list()
    for key in ('alt', 'ast', 'amylase', 'creatinine', 'degfr'):
        df = build_base_and_max(df, prefix=key)
    #     # Get trends
    #     columns = _columns_(key, df, temporal_columns_only=True)
    #     df[key + "_trend"] = build_easy_trend(df, columns)
    #
    #     # Get min / max levels
    #     columns = _columns_(key, df, temporal_columns_only=False)
    #     df[key + "_min"] = build_min(df, columns)
    #     df[key + "_max"] = build_max(df, columns)
    #
    #     columns_to_drop += columns
    # return df.drop(columns=columns_to_drop)
    return df

def replace(df, replacement_pairs, **_):
    """
    Replace older columns by new ones.

    Args:
        - df                : a DataFrame;
        - replacement_pairs : a dictionary to rename columns.

    Returns: the updated DataFrame.
    """
    replacement_pairs = [(old_col, new_col) for (old_col, new_col) in replacement_pairs if old_col in df.columns]
    for old_col, new_col in replacement_pairs:
        df = df.drop(columns=old_col) \
            .rename(columns={new_col: old_col})
    return df


def categorise(df, columns_to_categorise, **_):
    """
    Turn numerical columns into categorical ones.

    Args:
        - df                    : a DataFrame;
        - columns_to_categorise : a list of numerical columns to be turned into categorical ones.

    Returns: the updated DataFrame.
    """
    columns_to_drop = list()
    for key, cuts in columns_to_categorise.items():
        col = "cat_" + key
        df[col] = "before {0}".format(cuts[0])

        for i in range(1, len(cuts)):
            df[col] = df[col].mask(
                (cuts[i - 1] <= df[key]) & (df[key] < cuts[i]),
                "from {0} to {1}".format(cuts[i - 1], cuts[i])
            )

        df[col] = df[col].mask(cuts[-1] <= df[key], "after {0}".format(cuts[-1]))
        columns_to_drop.append(key)
    return df.drop(columns=columns_to_drop)

def adjust_prd_and_graft_no(df, **_):
    df["prd"]      = build_binary_prd(df)
    df["graft_no"] = build_binary_graft_no(df)
    return df

def impute_targets(df, **_):
    """
    Fix pcens/psurv and ecens/esurv columns

    Args:
        - df                  : a DataFrame.

    Returns: the updated DataFrame.
    """
    df['pcens'] = build_pcens(df)
    df['psurv'] = build_psurv(df)

    df['tcens'] = df.apply(build_tcens, axis=1)
    df['tsurv'] = df.apply(build_tsurv, axis=1)

    return df

def impute_multirisk(df, **_):
    """
    Build the competing risk columns from patient and graft survival data.

    Args:
        - df                  : a DataFrame.

    Returns: the updated DataFrame.
    """
    df['mcens'] = df.apply(build_mcens, axis=1)
    df['msurv'] = df.apply(build_msurv, axis=1)

    return df


def add_unknown_category(df, columns_with_unknowns, unknown, **_):
    """
    For selected columns, create an 'Unknown' category instead of introducing missing values.

    Args:
        - df                    : a DataFrame;
        - columns_with_unknowns : a list of columns that needs a specific treatment of their unknown values;
        - unknown               : the default, not np.nan/pd.NA, unknown value.

    Returns: the updated DataFrame.
    """
    df[columns_with_unknowns] = df[columns_with_unknowns].mask(
        df[columns_with_unknowns].isna(),
        unknown
    )

    return df


def remove_irrelevant_categories(df, irrelevant_categories, **_):
    """
    Drop rows containing categories tagged as irrelevant.

    Args:
        - df                    : a DataFrame;
        - irrelevant_categories : a list of categories that correspond to irrelevant rows (to discard).

    Returns: the updated DataFrame.
    """
    columns = intersect_columns(irrelevant_categories.keys(), df)

    for column in columns:
        df = df.drop(df[df[column].isin(irrelevant_categories[column])].index)

    return df


def remove_irrelevant_columns(df, descriptor, irrelevant_columns, **_):
    """
    Remove columns tagged as irrelevant.

    Args:
        - df                 : a DataFrame;
        - descriptor         : a Descriptor;
        - irrelevant_columns : a list of columns to discard.

    Returns: the updated DataFrame.
    """
    def _is_irrelevant_(column):
        try:
            return descriptor.get_entry(column).tags not in {"feature", "target"} \
                   or column in irrelevant_columns
        except KeyError:
            return True

    columns_to_drop = intersect_columns(
        [column for column in df.columns if _is_irrelevant_(column)],
        df
    )
    return df.drop(columns=columns_to_drop, errors='ignore')


def remove_constant_columns(df, **_):
    """
    Remove constant columns.

    Args:
        - df : a DataFrame.

    Returns: the updated DataFrame.
    """
    return df.drop(columns=get_constant_columns(df), errors='ignore')


# Reorder columns
def reorder_columns(df, descriptor, **_):
    """
    Re-order columns, with the targets at the end.

    Args:
        - df         : a DataFrame;
        - descriptor : a Descriptor.

    Returns: the updated DataFrame.
    """
    cols1 = [col for col in df.columns if descriptor.get_entry(col).tags != "target"]
    cols2 = [col for col in df.columns if descriptor.get_entry(col).tags == "target"]

    return df[sorted(cols1) + sorted(cols2)]


# The list of all the steps to perform.
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
    categorise,
    adjust_prd_and_graft_no,
    impute_targets,
    # impute_multirisk,
    replace,
    add_unknown_category,
    remove_irrelevant_categories,
    remove_irrelevant_columns,
    remove_constant_columns,
    reorder_columns
)


###################
#      CLEAN      #
###################


def clean_data(
    df,
    cleaning_steps      = CLEANING_STEPS,
    cleaning_parameters = None
):
    """
    Perform a list of cleaning steps on a DataFrame.

    Args:
        - df                  : the DataFrame to clean;
        - cleaning_steps      : the list of steps to perform;
        - cleaning_parameters : a dictionary containing the parameters relevant to the whole cleaning phase.

    Returns: the cleaned DataFrame.
    """
    clean_df = df.copy()

    for cleaning_step in cleaning_steps:
        clean_df = cleaning_step(clean_df, **cleaning_parameters)

    return clean_df

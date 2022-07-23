# Provide a suite of functions that create new columns from existing ones.

import numpy  as np
import pandas as pd

from project.data.describe   import Entry
from project.data.parameters import get_biolevel_columns,  \
                                    BMI_LIMITS,            \
                                    IRRELEVANT_CATEGORIES, \
                                    IRRELEVANT_COLUMNS,    \
                                    COLUMNS_WITH_UNKNOWNS, \
                                    UNKNOWN
from project.misc.clinical   import compute_bmi, compute_egfr
from project.misc.dataframes import density, intersect_columns


###############################
#      BUILD NEW COLUMNS      #
###############################


def build_egfr(input_df, creatinine_column):
    """
    For each row, compute the corresponding eGFR (depends on the creatinine column).
    """
    def f(df):
        return compute_egfr(df['dage'], df[creatinine_column], df['dsex'], df['dheight'])

    return input_df.apply(f, axis=1)


def build_max(df, columns):
    """
    For each row, give the maximum value among a set of columns.
    """
    return df[columns].max(axis=1)


def build_min(df, columns):
    """
    For each row, give the minimum value among a set of columns.
    """
    return df[columns].min(axis=1)


def build_easy_trend(df, columns):
    """
    Regarding a set of columns, tells for each row whether the minimum value is before or after the maximum value.
    """

    def f(column):
        if type(column) == str:
            return int(column[-2:])
        else:
            return np.nan

    return (df[columns].idxmax(axis=1).map(f) - df[columns].idxmin(axis=1).map(f)).apply(np.sign)


# from sklearn.linear_model import LinearRegression
#
# def build_trend(df, columns): TODO
#     """
#     For each row, describe the trend observed within a list of columns.
#     """
#     # gather all the compared values into one same list.
#     s = pd.Series(list(df[columns].values))
#
#     # Linear Regression
#     def f(L):
#         Y = np.array([y for y in L if not np.isnan(y)])
#         X = np.array([[x] for x in range(len(Y))])
#
#         if len(X) == 0:
#             return np.nan
#         elif len(X) == 1:
#             return 0
#         else:
#             reg = LinearRegression().fit(X, Y)
#
#             return reg.coef_[0]
#
#     s = s.apply(f)
#
#     #     t = "Stable"
#     #     t.mask(s.isna(), inplace=True)
#     #     t.mask(s - s.mean() >  s.std(), "Significant increase", inplace=True)
#     #     t.mask(s - s.mean() < -s.std(), "Significant decrease", inplace=True)
#
#     return s


def build_binary_code(df, columns, is_positive_or_negative):
    """
    Assuming that a set of columns can have their values either, positive, negative, or unknown,
    provides a base 3 representation of their values for each row.

    Args:
        - df                      : the input DataFrame
        - columns                 : the set of columns to consider
        - is_positive_or_negative : a dictionary mapping each column to a function that tells
                                    whether a known value is either postive or negative.
    Returns:
        A Series
    """
    s = pd.Series(0, index=df.index)

    for order, column in enumerate(columns):
        def f(x):
            if pd.isna(x):
                return x

            return is_positive_or_negative[column](x)

        binary_column = df[column].copy()
        binary_column = binary_column.apply(f)

        s.mask(binary_column == 0, s + 2 * (3 ** order), inplace=True)
        s.mask(binary_column == 1, s + 1 * (3 ** order), inplace=True)
    return s

#####################################
#      SELECT SPECIFIC COLUMNS      #
#####################################


def get_constant_columns(df):
    return df.columns[df.nunique() <= 1].to_list()


def get_irrelevant_columns(df, descriptor, additional_irrelevant_columns=IRRELEVANT_COLUMNS):
    def _is_irrelevant_(column):
        return descriptor.get_entry(column).tags not in {"feature", "target"} \
               or column in additional_irrelevant_columns
    return [column for column in df.columns if _is_irrelevant_(column)]


def get_sparse_columns(df, threshold):
    return [column for column in df.columns if density(df, column) < threshold]


#####################
#      WRANGLE      #
#####################

# Impute BMI
def impute_bmi(df, limits_bmi):
    min_threshold, max_threshold = limits_bmi

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
def transform_dialysis_columns(df, descriptor):
    is_positive_or_negative = {
        "days_on_dial_tx": lambda x: 0 if x > 0 else 1,
        "dial_at_reg"    : lambda x: 1 if x == "Not on dialysis" else 0,
        "dial_at_tx_type": lambda x: 1 if x == "Not on dialysis" else 0,
        "dial_at_tx"     : lambda x: descriptor.get_entry("dial_at_tx").binary_keys[x]
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
def recompute_egfr(df):
    creatinines = get_biolevel_columns('creatinine', df)
    egfrs       = get_biolevel_columns('degfr'     , df)
    creat_egfr  = zip(creatinines, egfrs)

    for creatinine, egfr in creat_egfr:
        s = build_egfr(df, creatinine)
        df[egfr] = df[egfr].where(s.isna(), s)
    return df

# Get trends, minimum, and maximum
def impute_biolevels(df):
    def _columns_(key_, df_, temporal_columns_only=False):
        columns_ = get_biolevel_columns(key_, df_, temporal_columns_only=temporal_columns_only)
        return intersect_columns(columns_, df_)

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
def add_unknown_category(df, columns_with_unknowns=COLUMNS_WITH_UNKNOWNS, unknown=UNKNOWN):
    df[columns_with_unknowns] = df[columns_with_unknowns].mask(df[columns_with_unknowns].isna(), unknown)

    return df
# Remove irrelevant_columns
def remove_irrelevant_categories(df, irrelevant_categories=IRRELEVANT_CATEGORIES):
    columns = intersect_columns(irrelevant_categories.keys(), df)

    for column in columns:
        df = df.drop(df[df[column].isin(irrelevant_categories[column])].index)

    return df

# Remove irrelevant columns
def remove_irrelevant_columns(df, descriptor, additional_irrelevant_columns=IRRELEVANT_COLUMNS):
    columns_to_drop = intersect_columns(get_irrelevant_columns(df, descriptor, additional_irrelevant_columns), df)
    return df.drop(columns=columns_to_drop)

# Remove constant columns
def remove_constant_columns(df):
    return df.drop(columns=get_constant_columns(df))

# Reorder columns
def reorder_columns(df, descriptor):
    cols1 = [col for col in df.columns if descriptor.get_entry(col).tags != "target"]
    cols2 = [col for col in df.columns if descriptor.get_entry(col).tags == "target"]

    return df[cols1 + cols2]

# Update Descriptor according to wrangling
def update_descriptor(descriptor, files="new"):
    """
    Update a descriptor by adding the new columns introduced with `wrangle_data`.
    """
    for key in ['alt', 'ast', 'amylase', 'creatinine', 'degfr']:
        descriptor.set_entry(Entry(
            key + "_trend",
            description="Trend for %s." % key,
            files=files,
            column_type="object",
            is_categorical=True,
            binary_keys="",
            tags="feature"
        ))

        descriptor.set_entry(Entry(
            key + "_min",
            description="Minimum value for %s." % key,
            files=files,
            column_type="float32",
            is_categorical=False,
            binary_keys="",
            tags="feature"
        ))

        descriptor.set_entry(Entry(
            key + "_max",
            description="Maximum value for %s." % key,
            files=files,
            column_type="float32",
            is_categorical=False,
            binary_keys="",
            tags="feature"
        ))

    descriptor.set_entry(Entry(
        "dial_type",
        description="Tells the most recent type of dialysis regarding transplantation.",
        files=files,
        column_type="object",
        is_categorical=True,
        binary_keys="",
        tags="feature"
    ))

    descriptor.set_entry(Entry(
        "dial_days",
        description="Tells how long the patient have been on dialysis.",
        files=files,
        column_type="float32",
        is_categorical=False,
        binary_keys="",
        tags="feature"
    ))

def wrangle_data(
        df,
        descriptor,
        limits_bmi=BMI_LIMITS,
        columns_with_unknowns=COLUMNS_WITH_UNKNOWNS,
        unknown=UNKNOWN,
        irrelevant_categories=IRRELEVANT_CATEGORIES,
        additional_irrelevant_columns=IRRELEVANT_COLUMNS,
        files="new"
):
    """
    Perform a more in-depth cleaning (wrangling) of the data.
    Assuming a basic cleaning of the data has already been performed, it:

    - Recompute the eGFR values;
    - Turn the columns related to biological levels into three simpler columns (for each level):
        - Minimum: the minimum observed value
        - Maximum: the maximum observed value
        - Trend  : a basic trend of the level
    - Remove constant and irrelevant columns
    - Recompute BMI values based on Height and Weight
    - Remove abnormal values based on recomputed values;
    - Simplify dialysis columns

    Args:
        - df         : an input DataFrame
        - descriptor : a Descriptor holding meta-data about the dataset
    Returns:
        An even cleaner version of the DataFrame provided as input.
    """
    wrangled_df = df.copy()
    update_descriptor(descriptor, files=files)

    # Impute BMI
    wrangled_df = impute_bmi(wrangled_df, limits_bmi=limits_bmi)

    # Transform dialysis related columns
    wrangled_df = transform_dialysis_columns(wrangled_df, descriptor)

    # Recompute eGFR
    wrangled_df = recompute_egfr(wrangled_df)

    # Get trends, minimum, and maximum
    wrangled_df = impute_biolevels(wrangled_df)

    # add unknown categories
    wrangled_df = add_unknown_category(wrangled_df, columns_with_unknowns, unknown)

    # Remove irrelevant categories
    wrangled_df = remove_irrelevant_categories(wrangled_df, irrelevant_categories)

    # Remove irrelevant columns
    wrangled_df = remove_irrelevant_columns(wrangled_df, descriptor, additional_irrelevant_columns)

    # Remove constant columns
    wrangled_df = remove_constant_columns(wrangled_df)

    # Reorder columns
    wrangled_df = reorder_columns(wrangled_df, descriptor)

    return wrangled_df

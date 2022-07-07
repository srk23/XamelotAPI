# Provide a suite of functions that create new columns from existing ones.

import numpy  as np
import pandas as pd

from project.data.clean      import get_constant_columns, get_irrelevant_columns
from project.data.variables  import get_biolevel_columns
from project.misc.clinical   import compute_bmi, compute_egfr
from project.misc.dataframes import intersect_columns

###############################
#      BIOLOGICAL LEVELS      #
###############################

def build_egfr(input_df, creatinine_column):
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


######################
#      DIALYSIS      #
######################


def build_binary_code(df, columns, is_positive_or_negative):
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


#####################
#      AUGMENT      #
#####################


def augment_data(df, descriptor):
    df = df.copy()

    # Recompute eGFR
    creatinines = get_biolevel_columns('creatinine', df)
    egfrs = get_biolevel_columns('degfr', df)
    creat_egfr = zip(creatinines, egfrs)

    for creatinine, egfr in creat_egfr:
        s = build_egfr(df, creatinine)
        df[egfr].where(s.isna(), s, inplace=True)

    # Get trends, minimum, and maximum
    for key in ['alt', 'ast', 'amylase', 'creatinine', 'degfr']:
        # Get trends
        columns = get_biolevel_columns(key, df, temporal_columns_only=True)
        columns = intersect_columns(columns, df)
        new_column = key + "_trend"
        df[new_column] = build_easy_trend(df, columns)

        # DESCRIPTOR.set_entry(Entry(
        #     new_column,
        #     description="Trend for %s." % key,
        #     files="new",
        #     column_type="object",
        #     is_categorical=True,
        #     binary_keys="",
        #     tags="feature"
        # ))

        # Get min / max levels
        columns = get_biolevel_columns(key, df)
        columns = intersect_columns(columns, df)

        new_column = key + "_min"
        df[new_column] = build_min(df, columns)
        # DESCRIPTOR.set_entry(Entry(
        #     new_column,
        #     description="Minimum value for %s." % key,
        #     files="new",
        #     column_type="float32",
        #     is_categorical=False,
        #     binary_keys="",
        #     tags="feature"
        # ))

        new_column = key + "_max"
        df[new_column] = build_max(df, columns)
        # DESCRIPTOR.set_entry(Entry(
        #     new_column,
        #     description="Maximum value for %s." % key,
        #     files="new",
        #     column_type="float32",
        #     is_categorical=False,
        #     binary_keys="",
        #     tags="feature"
        # ))

        df.drop(columns=columns, inplace=True)

    # Remove constant columns
    constant_columns = get_constant_columns(df)
    df.drop(constant_columns, axis=1, inplace=True)

    # Remove irrelevant columns
    irrelevant_columns = get_irrelevant_columns(df, descriptor)
    df.drop(irrelevant_columns, axis=1, inplace=True)

    # Impute BMI
    df['rbmi'] = df['rbmi'].mask(df['rbmi'].isna(), compute_bmi(df['rweight'], df['rheight']))
    df['dbmi'] = df['dbmi'].mask(df['dbmi'].isna(), compute_bmi(df['dweight'], df['dheight']))

    # Additional cleaning
    min_threshold = 5
    max_threshold = 60

    df['rweight'].where((min_threshold < df['rbmi']) & (df['rbmi'] < max_threshold), pd.NA, inplace=True)
    df['dweight'].where((min_threshold < df['dbmi']) & (df['dbmi'] < max_threshold), pd.NA, inplace=True)

    df['rheight'].where((min_threshold < df['rbmi']) & (df['rbmi'] < max_threshold), pd.NA, inplace=True)
    df['dheight'].where((min_threshold < df['dbmi']) & (df['dbmi'] < max_threshold), pd.NA, inplace=True)

    df['rbmi'].where((min_threshold < df['rbmi']) & (df['rbmi'] < max_threshold), pd.NA, inplace=True)
    df['dbmi'].where((min_threshold < df['dbmi']) & (df['dbmi'] < max_threshold), pd.NA, inplace=True)

    # Transform dialysis related columns
    is_positive_or_negative = {
        "days_on_dial_tx": lambda x: 0 if x > 0 else 1,
        "dial_at_reg": lambda x: 1 if x == "Not on dialysis" else 0,
        "dial_at_tx_type": lambda x: 1 if x == "Not on dialysis" else 0,
        "dial_at_tx": lambda x: descriptor.get_entry("dial_at_tx").binary_keys[x]
    }
    df["dial_code"] = build_binary_code(
        df,
        [
            "days_on_dial_tx",
            "dial_at_reg",
            "dial_at_tx_type",
            "dial_at_tx"
        ],
        is_positive_or_negative)

    df["dial_type"] = df["dial_at_tx_type"]
    df["dial_days"] = df["days_on_dial_tx"]

    # We adjust their values.
    df.loc[df['dial_code'].isin([3, 27, 30, 39, 41]), "dial_days"] = 0
    df.loc[df['dial_code'].isin([29, 32]), "dial_days"] = pd.NA

    df.loc[df['dial_code'].isin([3, 6, 8, 30, 33, 35, 44, 62]), "dial_type"] = df["dial_at_reg"]
    df.loc[df['dial_code'] == 27, "dial_type"] = "Not on dialysis"

    # We remove old colums.
    df = df.drop(['dial_at_reg', 'dial_at_tx', 'dial_at_tx_type', 'days_on_dial_tx', 'dial_code'], axis=1)

    return df

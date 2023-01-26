# Provide a set of functions that build new columns from existing ones.

import numpy  as np
import pandas as pd

from xmlot.misc.clinical import compute_egfr
from xmlot.misc.lists    import difference

def build_egfr(input_df, creatinine_column):
    """
    For each row, compute the corresponding eGFR (depends on the creatinine column).

    Args:
        - input_df          : a DataFrame;
        - creatinine_column : the column to focus on; note that age, sex, height can be hardcoded due to their unicity.

    Returns: the updated DataFrame.
    """
    def f(df):
        return compute_egfr(df['dage'], df[creatinine_column], df['dsex'], df['dheight'])

    return input_df.apply(f, axis=1)


def build_max(df, columns):
    """
    For each row, give the maximum value among a set of columns.

    Args:
        - df      : a DataFrame;
        - columns : a set of columns of df.

    Returns: the new column.
    """
    return df[columns].max(axis=1)


def build_min(df, columns):
    """
    For each row, give the minimum value among a set of columns.

    Args:
        - df      : a DataFrame;
        - columns : a set of columns of df.

    Returns: the new column.
    """
    return df[columns].min(axis=1)


def build_easy_trend(df, columns):
    """
    Regarding a set of columns, tells for each row whether the minimum value is before or after the maximum value.

    Args:
        - df      : a DataFrame;
        - columns : a set of columns of df.

    Returns: the new column.
    """

    def f(column):
        if type(column) == str:
            return int(column[-2:])
        else:
            return np.nan

    return (df[columns].idxmax(axis=1).map(f) - df[columns].idxmin(axis=1).map(f)).apply(np.sign)

def build_binary_code(df, columns, is_positive_or_negative):
    """
    Assuming that a set of columns can have their values either, positive, negative, or unknown,
    provides a base 3 representation of their values for each row.

    Args:
        - df                      : the input DataFrame
        - columns                 : the set of columns to consider
        - is_positive_or_negative : a dictionary mapping each column to a function that tells
                                    whether a known value is either postive or negative.
    Returns: a Series
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


def build_mcens(
        s,
        psurv="psurv",
        pcens="rdeath",
        gsurv="gsurv",
        gcens="gcens"
):
    """
    Based on patient survival time, patient death censoring, graft survival time, and graft censoring,
    allow to build the corresponding competing risks column.

    Args:
        - s     : a DataFrame's row (cf. pandas.apply)
        - psurv : column name for patient survival time;
        - pcens : column name for patient death censoring;
        - gsurv : column name for graft survival time;
        - gcens : column name for graft failure censoring.

    Returns: the corresponding competing risks censoring value for a given row.
    """
    if pd.notna(s[psurv]) \
            and pd.notna(s[gsurv]) \
            and pd.notna(s[pcens]) \
            and pd.notna(s[gcens]):
        # If graft event comes first,
        # or graft and death events are simultaneous,
        # but death event is censored:
        if s[gsurv] < s[psurv] \
                or (s[gsurv] == s[psurv] and not s[pcens]):
            if s[gcens]:  # Graft failure
                return "Graft failure"
            else:  # Censored
                return "Censored"
        else:  # Death gets priority
            if s[pcens]:
                return "Deceased"
            else:
                return "Censored"
    else:
        return "Unknown"


def build_msurv(s, columns=("psurv", "gsurv")):
    """
    Based on the patient survival time and the graft survival time,
    allow to build the corresponding competing survival time column
    by taking the minimum of these two.

    Args:
        - s      : a DataFrame's row (cf. pandas.apply)
        - columns: a set of column names

    Returns: the minimum of all the considered survival times for a given row.
    """
    psurv, gsurv = columns
    if pd.notna(s[psurv]) and pd.notna(s[gsurv]):
        return min(s[psurv], s[gsurv])
    return pd.NA

def build_classification_from_survival(
    df,
    t,
    accessor_code
):
    """
    Args:
        - df            : the input DataFrame;
        - t             : the span of time in which we want to predict the occurrence of events of interests.
        - accessor_code :

    Returns: a column that provides the corresponding labels for classification.
    """
    def _f_(_i_): return 0 if _i_ == 0 else _i_ + 1

    acc = getattr(df, accessor_code)

    # Initialisation
    s = pd.DataFrame(np.nan, index=df.index, columns=df.columns)[acc.event]

    # Get Alive labels
    s = s.mask(
        (acc.durations > t),
        other=1
    )

    # Get other labels
    for i, event in enumerate(difference(df[acc.event].value_counts().index, ["Unknown"])):
        s = s.mask(((acc.durations <= t) & (acc.events == event)), other=_f_(i))

    return s

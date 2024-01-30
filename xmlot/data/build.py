# Provide a set of functions that build new columns from existing ones.

import numpy  as np
import pandas as pd
import re

from xmlot.misc.clinical import compute_egfr
from xmlot.misc.lists    import difference


###############
#   TARGETS   #
###############


def build_pcens(
        df,
        pcens="pcens",
        censored="Censored"
):
    return df[pcens].fillna(censored)

def build_psurv(
        df,
        gsurv="gsurv",
        psurv="psurv"
):
    return df[psurv].fillna(df[gsurv])

def build_tcens(
        s,
        gcens="gcens",
        pcens="pcens",
        censored="Censored",
        event="Graft failure or death"
):
    if s[pcens] == censored and s[gcens] == censored:
        return censored
    else:
        return event


def build_tsurv(
        s,
        gcens="gcens",
        gsurv="gsurv",
        pcens="pcens",
        psurv="psurv",
        censored="Censored"
):
    if s[pcens] == censored and s[gcens] == censored:
        return min(s[gsurv], s[psurv])
    else:
        if s[gcens] != censored:
            return s[gsurv]

        if s[pcens] != censored:
            return s[psurv]

def build_mcens(
        s,
        psurv="psurv",
        pcens="pcens",
        gsurv="gsurv",
        gcens="gcens",
        censored="Censored",
        graft_failure="Graft failure",
        death="Deceased",
        unknown="Unknown"
):
    """
    Based on patient survival time, patient death censoring, graft survival time, and graft censoring,
    allow to build the corresponding competing risks column.

    Args:
        - s             : a DataFrame's row (cf. pandas.apply)
        - psurv         : column name for patient survival time;
        - pcens         : column name for patient death censoring;
        - gsurv         : column name for graft survival time;
        - gcens         : column name for graft failure censoring;
        - censored      : label for censoring;
        - graft_failure : label for graft failure;
        - unknown       : label for death.

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
                or (s[gsurv] == s[psurv] and s[pcens] == censored):
            if s[gcens] != censored:  # Graft failure
                return graft_failure
            else:  # Censored
                return censored
        else:  # Death gets priority
            if s[pcens] != censored:
                return death
            else:
                return censored
    else:
        return unknown


def build_msurv(s, psurv="psurv", gsurv="gsurv"):
    """
    Based on the patient survival time and the graft survival time,
    allow to build the corresponding competing survival time column
    by taking the minimum of these two.

    Args:
        - s      : a DataFrame's row (cf. pandas.apply)
        - psurv         : column name for patient survival time;
        - gsurv         : column name for graft survival time;

    Returns: the minimum of all the considered survival times for a given row.
    """
    if pd.notna(s[psurv]) and pd.notna(s[gsurv]):
        return min(s[psurv], s[gsurv])
    return pd.NA

def build_single_classification_from_survival(
    df,
    t,
    xcens,
    xsurv,
    events=None,
    unknown="Unknown",
    censored="Censored"
):
    """

    Args:
        - df       : the input DataFrame;
        - t        : the span of time in which we want to predict the occurrence of events of interests;
        - xcens    : column name for censoring;
        - xsurv    : column name for survival time;
        - unknown  : value to indicate an unknown event;
        - censored : value to indicate a censored event.

    Returns: a column that provides the corresponding labels for classification.
    """

    # Initialisation
    s = pd.DataFrame(np.nan, index=df.index, columns=df.columns)[xcens]

    # Get Alive labels
    s = s.mask(
        (df[xsurv] >= t),
        other=0
    )

    # Get other labels
    i_max = -1
    events = events if events is not None else difference(df[xcens].value_counts().index, [unknown, censored])
    for i, event in enumerate(events):
        i_max = max(i_max, i+1)
        s = s.mask(((df[xsurv] <= t) & (df[xcens] == event)), other=i + 1)

    # Deal with censored events
    s = s.mask(
        ((df[xsurv] < t) & (df[xcens] == censored)),
        other=i_max + 1
    )

    return s


def build_multi_classification_from_survival(
        df,
        t,
        gcens="gcens",
        gsurv="gsurv",
        pcens="pcens",
        psurv="psurv",
        censored="Censored"
):
    """
    Based on graft and patient survival, build the labels that correspond to patient states at a given time.
    These states can be: alive with functioning graft, alive with failed graft, deceased, and censored.

    Args:
        - df       : the input DataFrame;
        - t        : the span of time in which we want to predict the occurrence of events of interests;
        - gcens     : column name for graft censoring;
        - gsurv     : column name for graft survival time;
        - pcens     : column name for patient censoring;
        - psurv     : column name for patient survival time;
        - unknown  : value to indicate an unknown event;
        - censored : value to indicate a censored event.

    Returns: a column that provides the corresponding labels for classification.
    """
    #     alive_with_functioning_graft = "Alive with functioning graft",
    #     alive_with_failed_graft      = "Alive with failed graft",
    #     deceased                     = "Death of recipient",

    def _f_(_df_):

        graft_failure = (_df_[gcens] != censored)
        death = (_df_[pcens] != censored)

        if t < _df_[psurv]:
            if t < _df_[gsurv]:
                return 0  # alive_with_functioning_graft
            else:
                if graft_failure:
                    return 1  # alive_with_failed_graft
                else:
                    return 3  # censored
        else:
            if death:
                return 2  # deceased
            else:
                return 3  # censored

    return df.apply(_f_, axis=1)


def build_detailed_multi_classification_from_survival(
        df,
        t,
        gcens="gcens",
        gsurv="gsurv",
        pcens="pcens",
        psurv="psurv",
        censored="Censored"
):
    """
    Based on graft and patient survival, build the labels that correspond to patient states at a given time.
    These states can be: alive with functioning graft, alive with failed graft, deceased, and censored.

    Args:
        - df       : the input DataFrame;
        - t        : the span of time in which we want to predict the occurrence of events of interests;
        - gcens     : column name for graft censoring;
        - gsurv     : column name for graft survival time;
        - pcens     : column name for patient censoring;
        - psurv     : column name for patient survival time;
        - unknown  : value to indicate an unknown event;
        - censored : value to indicate a censored event.

    Returns: a column that provides the corresponding labels for classification.
    """

    def _f_(_df_):
        graft_failure = (_df_[gcens] != censored)
        failure_time  = _df_[gsurv]
        death         = (_df_[pcens] != censored)
        death_time    = _df_[psurv]

        alive_with_functioning_graft = 0
        alive_with_failed_graft      = 1
        dead_with_functioning_graft  = 2
        dead_with_failed_graft       = 3
        unknown_state                = 4

        if t < death_time:                     # the patient is still alive
            if t < failure_time:               # the graft is still functioning
                return alive_with_functioning_graft
            else:                              # the graft may have failed
                if graft_failure:              # the graft failed
                    return alive_with_failed_graft
                else:                          # the state of the graft is censored
                    return unknown_state
        else:                                  # the patient may have died
            if death:                          # the patient is dead
                if failure_time < death_time:  # in addition, the graft may have failed (before the death)
                    if graft_failure:          # ... with a failed graft
                        return dead_with_failed_graft
                    else:                      # we can't clearly say how the patient died
                        return unknown_state
                else:                          # but the graft didn't fail
                    return dead_with_functioning_graft
            else:                              # the state of the patient is censored
                return unknown_state

    return df.apply(_f_, axis=1)


################
#   FEATURES   #
################


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

    return (
            df[columns].idxmax(axis=1).map(f) - df[columns].idxmin(axis=1).map(f)
    ).apply(np.sign)


def build_base_and_max(df, prefix):
    def _f_(_df_):
        for i in [11, 12, 31, 32, 61, 62, 63]:
            col = prefix + "_" + str(i)
            if col in _df_.index and not pd.isna(_df_[col]):
                return _df_[prefix + "_" + str(i)]
            if i == 32 and prefix in _df_.index and not pd.isna(_df_[prefix]):
                return _df_[prefix]
        return np.nan
        # if not pd.isna(_df_[column + "_11"]):
        #     return _df_[column + "_11"]
        # elif not pd.isna(_df_[column + "_31"]):
        #     return _df_[column + "_31"]
        # elif not pd.isna(_df_[column]):
        #     return _df_[column]
        # elif not pd.isna(_df_[column + "_61"]):
        #     return _df_[column + "_61"]
        # elif not pd.isna(_df_[column + "_62"]):
        #     return _df_[column + "_62"]
        # elif not pd.isna(_df_[column + "_63"]):
        #     return _df_[column + "_63"]
        # else:
        #     pd.NA

    _df_ = df.copy()

    _df_[prefix + "_base"] = df.apply(_f_, axis=1)
    _df_[prefix + "_max"] = df[[col for col in df.columns if re.match(prefix, col)]].max(axis=1)

    _df_ = _df_.drop(columns=[
        col for col in _df_.columns if re.match(prefix + "($|_[0-9]+)", col)
    ])

    return _df_

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

def build_binary_prd(df):
    def _f_(s):
        if pd.isna(s):
            return np.nan
        elif s == "Diabetes":
            return "Diabetes"
        else:
            return "Other"
    return df["prd"].apply(_f_)

def build_binary_graft_no(df):
    def _f_(s):
        if s > 1:
            return "Second graft or more"
        else:
            return "First graft"
    return df["graft_no"].apply(_f_)

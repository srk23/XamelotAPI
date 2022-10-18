import pandas as pd

from project.misc.misc import get_var_name
from project.misc.list_operation import intersection

def build_empty_mask(df):
    return pd.DataFrame(False, index=df.index, columns=df.columns)


def density(df, column):
    return df[column].count() / len(df)


def intersect_columns(l, df):
    """
    Make sure that elements of the list `l` are columns of the DataFrame `df`.
    Args:
        - l  : a list supposed to contain columns;
        - df : a DataFrame where l's elements are supposed to come from.

    Returns:
         - a list: intersection between `l` and `df.columns` (follows `l`' ordering).
    """
    return intersection(l, df.columns)


def get_constant_columns(df):
    return df.columns[df.nunique() <= 1].to_list()

def get_sparse_columns(df, threshold):
    return [column for column in df.columns if density(df, column) < threshold]

#####################
#      COMPARE      #
#####################

class Comparison:
    def __init__(self, df1, df2, differences, depth=2):
        self.m_df1         = df1
        self.m_df2         = df2
        self.m_differences = differences
        self.m_depth       = depth

    @property
    def depth(self):
        return self.m_depth

    @depth.setter
    def depth(self, depth):
        self.m_depth = depth

    def __str__(self):
        name_df1 = get_var_name(self.m_df1, depth=self.m_depth)
        name_df2 = get_var_name(self.m_df2, depth=self.m_depth)

        string = "Comparing DataFrames df1={0} and df2={1}:\n".format(name_df1, name_df2)

        if self.m_differences == dict():
            string += "\n> They are equal."
        else:
            for k, v in self.m_differences.items():
                string += "\n> {0}:\n".format(k)
                for comment in v:
                    string += "\t- {0}\n".format(comment)

        return string

    def comments_count(self):
        count = dict()

        for comments in self.m_differences.values():
            for comment in comments:
                if comment not in count.keys():
                    count[comment] = 1
                else:
                    count[comment] += 1

        return count


def compare_dataframes(input_df1, input_df2, depth=2):
    """
    Compare two DataFrames.
    Args:
        - df1, df2  : the DataFrames to compare.

    Returns:
         - a dictionary: keys are mismatching columns, values are lists of comments about their differences.
    """
    df1 = input_df1.copy()
    df2 = input_df2.copy()

    difference = dict()
    if df1.equals(df2):
        return difference

    # Look for missing columns in one DataFrame or the other
    for column in (set(df1.columns) - set(df2.columns)):
        difference[column] = ["not in df2"]

    for column in (set(df2.columns) - set(df1.columns)):
        difference[column] = ["not in df1"]

    # Column-wise comparison
    def _append_message_(col, msg):
        if col in difference.keys():
            difference[col].append(msg)
        else:
            difference[col] = [msg]

    columns = intersect_columns(df1.columns, df2)
    for column in columns:
        # Check position
        if df1.columns.get_loc(column) != df2.columns.get_loc(column):
            _append_message_(column, "different order")

        # Check typing
        if df1[column].dtypes != df2.dtypes[column]:
            _append_message_(column, "different type")

        # Tries to retype before to continue comparison
        col1 = df1[column].dropna()
        col2 = df2[column].dropna().astype({column: col1.dtypes})

        # compare content
        if not col1.equals(col2):
            _append_message_(column, "different values")

    # Print

    return Comparison(df1, df2, difference, depth=depth)

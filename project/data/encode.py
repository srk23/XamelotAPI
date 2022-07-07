# Make sure DataFrames can be fed into machine learning models by one hot encoding categorical data.

import pandas as pd
import re

from project.data.describe      import Descriptor
from project.misc.miscellaneous import string_autotype


class OneHotEncoder:
    def __init__(self, descriptor: Descriptor, separator="#"):
        self.m_descriptor = descriptor
        self.m_sep        = separator

    @property
    def separator(self):
        return self.m_sep

    def is_encoded(self, column):
        return self.m_sep in column

    def split(self, column):
        return re.split(self.m_sep, column)

    def encode(self, df):
        encoded_df = df.copy()
        idx_cols = list()
        new_dfs = list()
        for column in encoded_df.columns:
            entry = self.m_descriptor.get_entry(column)
            if entry.is_categorical:
                if not entry.is_binary:
                    for category in encoded_df[column].value_counts().index:
                        new_column = column + self.m_sep + str(category)

                        # To prevent performance issues, we do not add encoded columns
                        # into encoded_df one by one;
                        # instead, we build separate DataFrames that we concatenate
                        # all at once at the end of this function.
                        new_df = pd.DataFrame([], columns=[], index=encoded_df.index)
                        new_df[new_column] = 0
                        new_df[new_column].mask(encoded_df[column] == category, other=1, inplace=True)
                        new_dfs.append(new_df)
                        idx_cols.append(new_column)
                    encoded_df.drop(columns=[column], inplace=True)
                else:
                    for k, v in entry.binary_keys.items():
                        encoded_df.loc[df[column] == k, column] = v
                    encoded_df[column] = encoded_df[column].astype('int64')
                    idx_cols.append(column)
            else:
                idx_cols.append(column)

        encoded_df = pd.concat([encoded_df, *new_dfs], axis=1)
        return encoded_df.reindex(idx_cols, axis=1)

    def decode(self, df):
        decoded_df = df.copy()
        columns_to_drop = list()
        idx_cols = list()

        for column in decoded_df.columns:
            if self.is_encoded(column):
                old_column, category = self.split(column)

                if old_column not in decoded_df.columns:
                    decoded_df[old_column] = pd.NA
                    idx_cols.append(old_column)

                adjust_type, cast_type = string_autotype(category)

                decoded_df[old_column].mask(decoded_df[column] == 1, other=adjust_type(category), inplace=True)
                decoded_df[old_column] = decoded_df[old_column].astype(cast_type, errors="ignore")

                columns_to_drop.append(column)
            else:
                entry = self.m_descriptor.get_entry(column)
                if entry.is_binary:
                    for k, v in entry.binary_values.items():
                        decoded_df.loc[df[column] == k, column] = v
                    decoded_df[column] = decoded_df[column].astype('Int64')

                idx_cols.append(column)

        decoded_df.drop(columns=columns_to_drop, inplace=True)
        return decoded_df.reindex(idx_cols, axis=1)

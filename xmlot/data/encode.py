# Allow to one hot encode categorical data.

import numpy as np
import pandas as pd
import re

from xmlot.data.dataframes import build_empty_mask
from xmlot.data.describe import Descriptor
from xmlot.misc.lists import difference


class OneHotEncoder:
    def __init__(
            self,
            descriptor: Descriptor,
            separator="#",
            exceptions=(),
            dummy=False,
            default_categories=None
    ):
        """
        Args:
            - descriptor         : a Descriptor to hold meta-data about the DataFrame to encode
            - separator          : a string token to separate column names and categories in one-hot encoded columns.
            - exceptions         : a list of columns to not encode.
            - dummy              : tells if dummy columns are considered or ignored/removed.
            - default_categories : a dictionary that maps a column to its default category
                                   (if not specified, that default category will be the first observed).
                                   We need to remember it during decoding without dummy columns.
        """
        self.m_descriptor = descriptor
        self.m_sep = separator
        self.m_dummy = dummy
        self.m_exceptions = exceptions
        self.m_default_categories = default_categories if default_categories is not None else dict()
        self.m_encoded_columns_lists = default_categories if default_categories is not None else dict()

        # Store the mapping between categories (especially binary ones) and their order.
        # Categorical keys  : category -> order    : needed in encode only, no need to be stored.
        # Categorical values: order    -> category : needed in decode and thus must be stored.

        self.m_categorical_values = dict()

        # Store column types for further decoding
        self.m_dtypes = None

    # Getters / Setters #

    @property
    def descriptor(self):
        return self.m_descriptor

    @property
    def separator(self):
        return self.m_sep

    @property
    def exceptions(self):
        return self.m_exceptions

    @property
    def dummy(self):
        return self.m_dummy

    @property
    def default_categories(self):
        return self.m_default_categories

    @property
    def columns(self):
        return self.m_dtypes.index

    # Miscellaneous #

    def is_encoded(self, column):
        return self.separator in column

    def split_column_name(self, column):
        return re.split(self.separator, column)

    # Encode / Decode #

    def encode(self, df, with_dummy_columns=False, reboot_encoded_columns=True):

        # Remark: dealing with dummy columns.
        #
        # In general, it is a good idea to omit one category in the encoding.
        # Indeed, a pure one hot encoding would induce some singularity in the data.
        # In other words, the corresponding matrix would not be invertible in that case.
        # Then, a given column can be seen as a linear combination of the other ones.

        encoded_df = df.copy()
        if pd.isna(encoded_df).any().any():
            encoded_df = encoded_df.replace(to_replace=pd.NA, value=np.nan)
        self.m_dtypes = df.dtypes

        idx_cols = list()
        new_dfs = list()

        if reboot_encoded_columns:
            self.m_encoded_columns_lists = dict()
            self.m_dummy                 = with_dummy_columns

        for column in encoded_df.columns:
            entry = self.descriptor.get_entry(column)

            # One-hot encode if...
            if entry.is_categorical \
                    and column not in self.exceptions \
                    and (not entry.is_binary or self.dummy):

                if reboot_encoded_columns:
                    self.m_encoded_columns_lists[column] = list()

                # Ensure there is a default category (if not, build it):
                # it will be dropped if removing dummy columns is required.
                if column not in self.default_categories.keys():
                    first_category = df[column].value_counts().index[0]
                    self.default_categories[column] = str(first_category)

                # If required, add a dummy column: use the default category for that.
                dummy_column = column + self.separator + self.default_categories[column]
                if self.dummy:
                    encoded_df[dummy_column] = 1
                    idx_cols.append(dummy_column)

                    if reboot_encoded_columns:
                        self.m_encoded_columns_lists[column].append(self.default_categories[column])

                # Process each category, in order of importance.
                if reboot_encoded_columns:
                    categories = df[column].value_counts().index
                else:
                    categories = self.m_encoded_columns_lists[column]
                for category in categories:
                    # If the category is not default.
                    if str(category) != self.default_categories[column]:
                        new_column = column + self.separator + str(category)

                        # To prevent performance issues, we do not add encoded columns
                        # into encoded_df one by one;
                        # instead, we build separate DataFrames that we concatenate
                        # all at once at the end of this function.

                        new_df = pd.DataFrame([], columns=[], index=encoded_df.index)
                        new_df[new_column] = 0
                        new_df[new_column] = new_df[new_column].mask(df[column] == category, other=1)
                        new_dfs.append(new_df)
                        idx_cols.append(new_column)

                        if reboot_encoded_columns:
                            self.m_encoded_columns_lists[column].append(str(category))

                        # Update encoded_df[dummy_column] if considering dummy columns.
                        if self.dummy:
                                encoded_df[dummy_column] = encoded_df[dummy_column] - new_df[new_column]
                # Drop the old categorical column once processed
                encoded_df = encoded_df.drop(columns=[column])

            # If the column is categorical but do not require to be encoded:
            elif (entry.is_categorical \
                  and (column in self.exceptions or (entry.is_binary and not self.dummy))):
                if entry.categorical_keys is not None:
                    categorical_keys = entry.categorical_keys
                else:
                    categorical_keys = {k: v for (v, k) in enumerate(encoded_df[column].value_counts().index)}
                    self.m_categorical_values[column] = {
                        v: k for (v, k) in enumerate(encoded_df[column].value_counts().index)
                    }

                def _replace_categories_(s):
                    try:
                        return categorical_keys[s[column]]
                    except KeyError:
                        return np.nan

                encoded_df[column] = encoded_df[[column]].apply(_replace_categories_, axis=1)
                idx_cols.append(column)

            else:  # ... do not encode.
                idx_cols.append(column)

        # Concat the columns all together
        encoded_df = pd.concat([encoded_df, *new_dfs], axis=1)

        # Check for forgotten Nan and adjust.
        condition = build_empty_mask(encoded_df)

        for column, categories in self.m_encoded_columns_lists.items():
            if column in df.columns:
                condition_per_column = df[column].isna()
                for category in categories:
                    encoded_column = column + self.separator + str(category)
                    condition[encoded_column] = condition_per_column
        encoded_df = encoded_df.mask(condition, other=np.nan)

        # Finally, re-order them, and uniformise types to `float32`.
        return encoded_df.reindex(idx_cols, axis=1).astype('float32')

    def decode(self, df, warning_on=True):
        decoded_df = df.copy()

        columns_to_drop = list()
        columns_to_add  = list()
        idx_cols = list()

        for column in df.columns:
            if self.is_encoded(column):
                old_column, category = self.split_column_name(column)

                # If it is the first time the prefix is observed,
                if old_column not in idx_cols:
                    idx_cols.append(old_column)
                    columns_to_drop.append(column)
                    columns = [col for col in df.columns if re.match(old_column, col) is not None]

                    partial_df = df[columns].copy()

                    # if a dummy column has been removed, we add it.
                    if not self.dummy:
                        default_cat = self.default_categories[old_column]

                        s = partial_df.sum(axis=1)

                        partial_df[old_column + self.separator + default_cat] = 1 - s

                    partial_df.columns = [col[len(old_column)+len(self.separator):] for col in partial_df.columns]

                    # Any row containing only nan
                    partial_df = partial_df.mask(df[columns].isnull().all(axis=1), other=np.nan)

                    # Any complete encoding for which the sum is zero, leads to an unknown value.
                    partial_df = partial_df.mask((partial_df.sum(axis=1) == 0), other=np.nan)

                    # Decode one hot encoding by taking the argmax.
                    partial_df = pd.DataFrame(partial_df.idxmax(axis=1, skipna=True), columns=[old_column])

                    # Insert unknown values when in presence of inconsistent encodings.
                    mask = (df[columns].copy().sum(axis=1) > 1)
                    if sum(mask) > 0:
                        if warning_on:
                            print("WARNING: some one hot encodings for column '{}' sums above one!".format(old_column))
                        partial_df = partial_df.mask(mask, other=pd.NA)

                    columns_to_add.append(partial_df)

                #     decoded_df[old_column] = np.nan  # self.default_categories[old_column]
                #     idx_cols.append(old_column)
                #
                # # Replace unknown values based on the encoded columns
                # decoded_df[old_column] = decoded_df[old_column].mask(df[column] == 1, other=category)
                # columns_to_drop.append(column)
                #
                # # If the dummy columns are not here, we need to add the default values for the categorical columns.
                # entry = self.descriptor.get_entry(old_column)
                # if entry.is_categorical and not entry.is_binary:
                #     # Get the list of all the encoded columns related to old_column
                #     r = re.compile("^" + old_column)
                #     encoded_columns_list = list(filter(r.match, df.columns))
                #
                #     # Mask rows with only zeros with default category.
                #     decoded_df[old_column] = decoded_df[old_column].mask(
                #         (df[encoded_columns_list].sum(axis=1) == 0) & ~df[encoded_columns_list].isna().any(axis=1),
                #         other=self.default_categories[old_column]
                #     )

            else:  # ... the column has not been one hot encoded:
                # it is either numerical, binary (in a "no dummy" case), or an exception.
                entry = self.m_descriptor.get_entry(column)
                if entry.is_numerical:  # Nothing is required
                    pass
                else:  # The column is either binary or an exception
                    if entry.categorical_values is not None:
                        categorical_values = entry.categorical_values
                    else:
                        categorical_values = self.m_categorical_values[column]

                    def _restore_categories_(_s_):
                        try:
                            return categorical_values[_s_[column].round()]
                        except KeyError:
                            return np.nan

                    print("WARNING: column '{}' may be rounded to avoid conversion error.".format(column))
                    decoded_df[column] = decoded_df[[column]].apply(_restore_categories_, axis=1)
                idx_cols.append(column)

        decoded_df = decoded_df.drop(columns=columns_to_drop)
        decoded_df = pd.concat([decoded_df] + columns_to_add, axis=1)
        decoded_df = decoded_df.reindex(idx_cols, axis=1)

        index_to_drop = difference(pd.DataFrame(self.m_dtypes).transpose().columns, decoded_df.columns)
        dtypes = self.m_dtypes.drop(index=index_to_drop)

        int_64_cols = dtypes[dtypes == "Int64"].index
        if len(int_64_cols) > 0:
            print("WARNING: columns {} are rounded to avoid conversion error.".format(list(int_64_cols)))
            decoded_df[int_64_cols] = decoded_df[int_64_cols].round()

        return decoded_df.astype(dtypes)

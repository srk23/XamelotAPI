# Functions to transform/scale (e.g. standardisation, normalisation) numerical data.

import re

def get_standardisation(df):
    return {
        "get_center" : lambda col: df[col].mean(),
        "get_scale"  : lambda col: df[col].std()
    }

def get_normalisation(df):
    return {
        "get_center" : lambda col: df[col].min(),
        "get_scale"  : lambda col: df[col].max() - df[col].min()
    }

class Scaler:
    def __init__(
            self,
            df,
            accessor_code,
            ohe,
            get_center=lambda col: 0,
            get_scale =lambda col: 1,
    ):
        descriptor = ohe.descriptor
        separator  = ohe.separator

        self.m_columns_to_transform = list()
        for column in getattr(df, accessor_code).features_list:
            if not re.findall(separator, column):
                if descriptor.get_entry(column).is_numerical:
                    self.m_columns_to_transform.append(column)

        self.m_centers = {column: get_center(column) for column in self.m_columns_to_transform}
        self.m_scales  = {column: get_scale(column)  for column in self.m_columns_to_transform}

        assert 0 not in self.m_scales.values(), "One or more columns are constant, inducing a division by zero."

    def __call__(self, df):
        def _transform_(s):
            return (s - self.m_centers[s.name]) / self.m_scales[s.name]

        output_df = df.copy()
        output_df[self.m_columns_to_transform] = output_df[self.m_columns_to_transform].apply(_transform_)

        return output_df

    def undo(self, df):
        """
        From a scaled DataFrame, reverts to its original form.
        """
        def _untransform_(s):
            return (s * self.m_scales[s.name]) + self.m_centers[s.name]

        output_df = df.copy()
        output_df[self.m_columns_to_transform] = output_df[self.m_columns_to_transform].apply(_untransform_)
        return output_df

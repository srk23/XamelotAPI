# Provide functions to split data.

import numpy  as np
import pandas as pd


def split_dataset(df, fracs, main_target, random_states=(None, None)):
    """
    Split data in a stratified way.
    Stratification is done regarding a column given as main target:
    the label proportions in that column are preserverd across splits.
    """
    df_ = df.copy()
    counts = df_[main_target].value_counts()

    splits = list()

    for frac in fracs:
        dfs = list()
        for value in counts.index:
            n = int(np.floor(frac * counts[value]))

            extracted_df = df_.loc[df_[main_target] == value, :].sample(n=n, random_state=random_states[0])
            dfs.append(extracted_df)
        split_df = pd.concat(dfs).sample(frac=1, random_state=random_states[1])

        df_ = df_.drop(index=split_df.index)

        splits.append(split_df)

    return splits

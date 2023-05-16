# Provide functions to split data.

import numpy  as np
import pandas as pd
from xmlot.misc.misc import SeedGenerator

def split_dataset(df, fracs, main_target=None, seed=None):
    """
    Split data in a stratified way.
    Stratification is done regarding a column given as main target:
    the label proportions in that column are preserverd across splits.
    """
    seed_generator = SeedGenerator(seed)

    df_ = df.copy()
    counts = df_[main_target].value_counts() if main_target is not None else None

    splits = list()

    for frac in fracs:

        if counts is None:
            n = int(np.floor(frac * len(df_)))
            split_df = df_.sample(n=n, random_state=seed_generator())
        else:
            dfs = list()
            for value in counts.index:
                n = int(np.floor(frac * counts[value]))

                df_n = df_.loc[df_[main_target] == value, :]
                extracted_df = df_n.sample(n=min(n, len(df_n)), random_state=seed_generator())
                dfs.append(extracted_df)

            split_df = pd.concat(dfs).sample(frac=1, random_state=seed_generator())

            df_ = df_.drop(index=split_df.index)

        splits.append(split_df)

    return splits

def sample_dataset(df, n, main_target, seed=None):
    N = len(df)
    return split_dataset(df, fracs=[n / N, (N - n) / N], main_target=main_target, seed=seed)[0]

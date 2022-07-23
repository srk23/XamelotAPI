import numpy  as np
import pandas as pd

def split_dataset(df, fracs, main_target):
    df_ = df.copy()
    counts = df_[main_target].value_counts()

    splits = list()

    for frac in fracs:
        dfs = list()
        for value in counts.index:
            n = int(np.floor(frac * counts[value]))

            extracted_df = df_.loc[df_[main_target] == value, :].sample(n=n)
            dfs.append(extracted_df)
        split_df = pd.concat(dfs).sample(frac=1)
        
        df_ = df_.drop(index=split_df.index)

        splits.append(split_df)

    return splits

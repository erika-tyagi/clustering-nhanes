import numpy as np
import pandas as pd

def check_any_nan_or_inf(df, name='df'):
    """
    Checks a dataframe to see if it contains NaN or infinity values.
    """
    total = len(df)

    for colname in df.columns:
        if np.isnan(df[colname]).any():
            count = sum(np.isnan(df[colname]))
            ratio = count / total
            raise Exception(f"Found NaN in column {colname} in {name} ({ratio}).")
        elif not np.isfinite(df[colname].values).all():
            count = sum(np.isfinite(df[colname]))
            ratio = (total - count) / total
            raise Exception(f"Found Inf in column {colname} in {name} ({ratio}).")

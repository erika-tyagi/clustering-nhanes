import numpy as np
import pandas as pd

DEFAULTS = {
    'label_col': 'label',
    'evictions_col': 'evictions',
    'drop_column': True
}

def label(split, lower_bound, **kwargs):
    """
    Labels a train/test df pair.
    """
    train_df, test_df = split
    train_df = label_df(train_df, lower_bound, **kwargs)
    test_df = label_df(test_df, lower_bound, **kwargs)
    return train_df, test_df


def label_df(df, lower_bound, **kwargs):
    """
    Labels rows that have evictions above the given lower bound as a 1,
    while rows below that bound get a 0.
    """
    settings = _settings(kwargs)
    label_col = settings['label_col']
    evictions_col = settings['evictions_col']
    drop_column = settings['drop_column']

    df = df.copy() # Err, think about why this is needed later

    df[label_col] = np.where(df[evictions_col] > lower_bound, 1, 0)

    if drop_column:
        df = df.drop(columns=[evictions_col])

    return df


def label_df_pct(df, pct, sort_by_var, **kwargs):
    settings = _settings(kwargs)
    label_col = settings['label_col']

    df = df.copy() # Err, think about why this is needed later
    df = df.sort_values(sort_by_var, ascending=False)

    cutoff_index = int(len(df) * (pct / 100.0))
    labels = [1 if i < cutoff_index else 0 for i in range(len(df))]
    df[label_col] = labels
    return df


def _settings(kwargs):
    settings = DEFAULTS.copy()
    settings.update(kwargs)
    return settings

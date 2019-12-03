import pandas as pd


def calc_summary_stats(df, cols_to_drop=None, cols_to_include=None):
    '''
    Calculates mean, median, standard deviation, max and min values for all columns
    (note: this may be non-sensical for categorical variables, so optional parameters
    allow for specification of either which columns should remain or which should
    be eliminated)
    '''

    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)

    if cols_to_include:
        df = df[cols_to_include]

    summary_df = pd.DataFrame(df.mean())
    summary_df = summary_df.rename(columns={0:'mean'})
    summary_df['std_dev'] = df.std()
    summary_df['median'] = df.median()
    summary_df['max_val'] = df.max()
    summary_df['min_val'] = df.min()

    return summary_df

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def highlight_min(data, color='yellow'):
    '''
    highlight the minimum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)

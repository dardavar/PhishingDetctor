import numpy as np
import pandas as pd


# assume uniform distribution
def flatten_col(col, max_values=5):
    if len(set(col)) < max_values:
        return col
    else:
        values_range_length = col.max() - col.min()
        return np.ceil(max_values * (col / values_range_length))


def flatten_df(df, max_values=5):
    return df.apply(flatten_col)


def calc_entropy(series):
    return max(0, -1 * sum([p*np.log2(p) for p in series.value_counts(normalize=True)]))


def get_best_feature_to_split_by(df, data_entropy, main_col_name):
    df_size = len(df)
    best_feature_to_split_by = ''
    best_mutual_information = 0
    for col in df.columns:
        if col == main_col_name:
            continue
        entropy_per_value = df[[col, main_col_name]].groupby(col)[main_col_name].agg([calc_entropy, 'count'])
        entropy_per_value['normalized_entropy'] = \
            (entropy_per_value['count']/df_size) * entropy_per_value['calc_entropy']

        mutual_information = data_entropy - entropy_per_value['normalized_entropy'].sum()

        if mutual_information > best_mutual_information:
            best_mutual_information = mutual_information
            best_feature_to_split_by = col

    return best_feature_to_split_by


def is_successful(row):
    return row['result'] == row['status']


def is_false_positive(row):
    return row['result'] == 0 and row['status'] == 1


def is_false_negative(row):
    return row['result'] == 1 and row['status'] == 0

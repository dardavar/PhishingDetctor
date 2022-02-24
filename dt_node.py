import numpy as np
from utils import get_best_feature_to_split_by, calc_entropy


class DTNode:

    def __init__(self, df, main_col_name, feature_name=None, feature_value=None):
        self.df = df
        self.feature_name = feature_name
        self.feature_value = feature_value
        self.num_of_ones = len(df[df[main_col_name] == 1])
        self.num_of_zeroes = len(df[df[main_col_name] == 0])
        self.node_entropy = calc_entropy(df[main_col_name])
        self.next_feature_name = None
        self.is_pure = self.num_of_zeroes == 0 or self.num_of_ones == 0
        self.is_indecisive = self.node_is_indecisive(df)
        self.next_nodes = self.generate_next_nodes(df, main_col_name)

    # it is possible to get node with indistinct value, we need to check that
    def node_is_indecisive(self, df):
        return len(df.columns) == 1 and not self.is_pure

    def generate_next_nodes(self, df, main_col_name):
        if self.is_pure or self.is_indecisive:
            return {}
        else:
            next_nodes = {}
            feature_to_split_by = get_best_feature_to_split_by(df, self.node_entropy, main_col_name)
            self.next_feature_name = feature_to_split_by
            for value in set(df[feature_to_split_by]):
                this_name_df = df[df[feature_to_split_by] == value]
                next_nodes[value] = DTNode(
                    df=this_name_df.drop(feature_to_split_by, axis=1),
                    main_col_name=main_col_name,
                    feature_name=feature_to_split_by,
                    feature_value=value)
            return next_nodes

    def test_data(self, row):
        def find_nearest(array, val):
            array = np.asarray(array)
            idx = (np.abs(array - val)).argmin()
            return array[idx]
        if self.is_indecisive:
            return -1
        elif self.is_pure:
            if self.num_of_zeroes != 0:
                return 0
            else:
                return 1
        else:
            # this is a code that needs to be fixed - it caused by a missing partition in the flat df
            value = row[self.next_feature_name]
            nodes_values = list(self.next_nodes.keys())
            return self.next_nodes[find_nearest(nodes_values, value)].test_data(row)






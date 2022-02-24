import pandas as pd
import argparse
from utils import flatten_df, is_false_positive, is_false_negative, is_successful
from dt_node import DTNode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn to identify phishing')
    parser.add_argument('path_to_csv', type=str)
    parser.add_argument('main_col_name', type=str, nargs='?')
    parser.add_argument('url_col_name', type=str, nargs='?')
    parser.add_argument('training_size', type=int, nargs='?')
    parser.add_argument('testing_size', type=int, nargs='?')

    args = parser.parse_args()
    df = pd.read_csv(args.path_to_csv).drop(args.url_col_name or 'url', axis=1)
    df = flatten_df(df, 5)

    training_df = df.head(args.training_size or 1000)
    testing_df = df.tail(args.testing_size or 1000)

    main_col_name = args.main_col_name or 'status'

    # generate tree
    root = DTNode(training_df, main_col_name, None)

    testing_df['result'] = testing_df.apply(root.test_data, axis=1)

    testing_df = testing_df.apply([is_successful, is_false_positive, is_false_negative], axis=1)

    total_tests = len(testing_df)
    success_rate = testing_df['is_successful'].sum() / total_tests
    false_positive_rate = testing_df['is_false_positive'].sum() / total_tests
    false_negative_rate = testing_df['is_false_negative'].sum() / total_tests

    print("success rate is: {}\nfalse positive rate is: {}\nfalse negative rate is: {}"
          .format(success_rate, false_positive_rate, false_negative_rate))




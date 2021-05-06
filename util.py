from sklearn.model_selection import train_test_split
import numpy as np


def generate_train_test_split(df):
    """
    Generate train test split
    :param df: Dataset to split
    :type df: DataFrame
    :return: Returns two datasets
    :rtype: DataFrame, DataFrame
    """

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    return train, test


def generate_train_dev_test_split(df):
    """
    Generate train dev test split
    :param df: Dataset to split
    :type df: DataFrame
    :return: Returns three datasets
    :rtype: DataFrame, DataFrame, DataFrame
    """

    # Split into 60% train, 20% dev and 20% test
    train, dev, test = np.split(df.sample(frac=1, random_state=42),
                                [int(.6 * len(df)), int(.8 * len(df))])

    return train, dev, test

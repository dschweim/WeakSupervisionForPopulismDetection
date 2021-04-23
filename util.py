import pandas as pd
from sklearn.model_selection import train_test_split


def generate_train_test_split(df):
    """
    Generate train test split
    :return:
    :rtype:
    """

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    return train, test



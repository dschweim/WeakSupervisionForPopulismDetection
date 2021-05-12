from sklearn.model_selection import train_test_split
import numpy as np
import re


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


def standardize_party_naming(party):
    """
    Standardize party naming
    :param party: String with party name
    :type party:  str
    :return: Pre-processed string with party name
    :rtype:  str
    """

    if party is None:
        return None
    else:
        # Lowercase
        party = party.lower()
        # Remove whitespaces
        party = party.strip()

        # Remove umlauts
        party = party.replace('ö', 'o')
        party = party.replace('ü', 'u')
        party = party.replace('ä', 'a')

        # Standardize naming of specific parties
        if re.search(r'stronach', party):
            party = 'teamstronach'

        if re.search(r'linke', party):
            party = 'dielinke'

        if re.search(r'grunen', party):
            party = 'grune'

        return party

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
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

        # Standardize naming of "Team Stronach"
        if re.search(r'stronach', party):
            party = 'teamstronach'

        return party


def get_lfs_external_inputs(data_path: str):
    """
     Generate Lists of pop and nonpop parties for each CHES study
    :param data_path: Dictionary with further input for labeling functions
    :type data_path:  str
    :return: Path to data folder
    :rtype:  dict
    """
    # Generate binary-labelled POP dataframes for Austrian, German and Swiss parties
    rel_countries = ['aus', 'ger', 'swi']
    rel_country_ids = ['13', '3', '36']


    # CHES 14:
    ches_df_14 = pd.read_csv(f'{data_path}\\CHES\\2014_CHES_dataset_means.csv')
    ches_df_14 = ches_df_14.loc[ches_df_14.country.isin(rel_country_ids)]
    ches_df_14['antielite_salience'] = ches_df_14['antielite_salience'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_14 = ches_df_14[['party_name', 'antielite_salience']]
    # Rename "party_name" col
    ches_df_14.rename(columns={"party_name": "party"}, inplace=True)

    # Split entries for parties with two naming conventions to separate entries
    add_entries_14 = pd.DataFrame()
    for index, row in ches_df_14.iterrows():
        # Find entries with two party namings (such as 'EDU/UDF')
        match = re.search(r'([^\/]+)\/([^\/]+)', row.party)
        if match:
            # Make first party the name of current row
            ches_df_14.loc[index, 'party'] = match.group(1)
            # Construct new row with second name and copy values for other cols
            entry = pd.DataFrame({'party': [match.group(2)],
                                  'antielite_salience': [row.antielite_salience]
                                  })
            # Add row to df which will be appended at the end
            add_entries_14 = add_entries_14.append(entry)

    # Add additional rows
    ches_df_14 = ches_df_14.append(add_entries_14)

    # Standardize party naming: lowercase, remove Umlaute, remove empty spaces, etc
    ches_df_14['party'] = ches_df_14['party'].apply(lambda x: standardize_party_naming(x))

    ches_14_pop = \
        ches_df_14.loc[(ches_df_14.antielite_salience == 1)]['party'].tolist()
    ches_14_nonpop = \
        ches_df_14.loc[(ches_df_14.antielite_salience == 0)]['party'].tolist()
    ches_14 = {'pop': ches_14_pop,
               'nonpop': ches_14_nonpop}


    # CHES 17:
    ches_df_17 = pd.read_csv(f'{data_path}\\CHES\\CHES_means_2017.csv')
    ches_df_17 = ches_df_17.loc[ches_df_17.country.isin(rel_countries)]
    ches_df_17['people_vs_elite'] = ches_df_17['people_vs_elite'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_17['antielite_salience'] = ches_df_17['antielite_salience'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_17 = ches_df_17[['party', 'people_vs_elite', 'antielite_salience']]

    # Split entries for parties with two naming conventions to separate entries
    add_entries_17 = pd.DataFrame()
    for index, row in ches_df_17.iterrows():
        # Find entries with two party namings (such as 'EDU/UDF')
        match = re.search(r'([^\/]+)\/([^\/]+)', row.party)
        if match:
            # Make first party the name of current row
            ches_df_17.loc[index, 'party'] = match.group(1)
            # Construct new row with second name and copy values for other cols
            entry = pd.DataFrame({'party': [match.group(2)],
                                  'people_vs_elite': [row.people_vs_elite],
                                  'antielite_salience': [row.antielite_salience]
                                  })
            # Add row to df which will be appended at the end
            add_entries_17 = add_entries_17.append(entry)

    # Add additional rows
    ches_df_17 = ches_df_17.append(add_entries_17)

    # Standardize party naming: lowercase, remove Umlaute, remove empty spaces, etc
    ches_df_17['party'] = ches_df_17['party'].apply(lambda x: standardize_party_naming(x))

    # Generate list of pop and nonpop parties from CHES 17
    ches_17_pop = \
        ches_df_17.loc[(ches_df_17.people_vs_elite == 1) | (ches_df_17.antielite_salience == 1)]['party'].tolist()
    ches_17_nonpop = \
        ches_df_17.loc[(ches_df_17.people_vs_elite == 0) & (ches_df_17.antielite_salience == 0)]['party'].tolist()
    ches_17 = {'pop': ches_17_pop,
               'nonpop': ches_17_nonpop}


    # CHES 19:
    ches_df_19 = pd.read_csv(f'{data_path}\\CHES\\CHES2019V3.csv')
    ches_df_19 = ches_df_19.loc[ches_df_19.country.isin(rel_country_ids)]
    ches_df_19['people_vs_elite'] = ches_df_19['people_vs_elite'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_19['antielite_salience'] = ches_df_19['antielite_salience'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_19 = ches_df_19[['party', 'people_vs_elite', 'antielite_salience']]

    # Split entries for parties with two naming conventions to separate entries
    add_entries_19 = pd.DataFrame()
    for index, row in ches_df_19.iterrows():
        # Find entries with two party namings (such as 'EDU/UDF')
        match = re.search(r'([^\/]+)\/([^\/]+)', row.party)
        if match:
            # Make first party the name of current row
            ches_df_19.loc[index, 'party'] = match.group(1)
            # Construct new row with second name and copy values for other cols
            entry = pd.DataFrame({'party': [match.group(2)],
                                  'people_vs_elite': [row.people_vs_elite],
                                  'antielite_salience': [row.antielite_salience]
                                  })
            # Add row to df which will be appended at the end
            add_entries_19 = add_entries_19.append(entry)

    # Add additional rows
    ches_df_19 = ches_df_19.append(add_entries_19)

    # Standardize party naming: lowercase, remove Umlaute, remove empty spaces, etc
    ches_df_19['party'] = ches_df_19['party'].apply(lambda x: standardize_party_naming(x))

    # Generate list of pop and nonpop parties from CHES 19
    ches_19_pop = \
        ches_df_19.loc[(ches_df_19.people_vs_elite == 1) | (ches_df_19.antielite_salience == 1)]['party'].tolist()
    ches_19_nonpop = \
        ches_df_19.loc[(ches_df_19.people_vs_elite == 0) & (ches_df_19.antielite_salience == 0)]['party'].tolist()
    ches_19 = {'pop': ches_19_pop,
               'nonpop': ches_19_nonpop}

    return ches_14, ches_17, ches_19

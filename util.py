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

        # Standardize naming of specific parties
        if re.search(r'stronach', party):
            party = 'teamstronach'

        if re.search(r'linke', party):
            party = 'dielinke'

        if re.search(r'grunen', party):
            party = 'grune'

        return party


def get_lfs_external_inputs(data_path: str):
    """
     Generate Lists of pop and nonpop parties for each CHES study
    :param data_path: Dictionary with further input for labeling functions
    :type data_path:  str
    :return: dict_ches_14, dict_ches_17, dict_ches_19
    :rtype:  dict
    """
    # Generate binary-labelled POP dataframes for Austrian, German and Swiss parties
    rel_countries = ['aus', 'ger', 'swi']
    rel_country_ids = ['13', '3', '36']


    # CHES 14:
    ches_df_14 = pd.read_csv(f'{data_path}\\CHES\\2014_CHES_dataset_means.csv')
    ches_df_14 = ches_df_14.loc[ches_df_14.country.isin(rel_country_ids)]
    ches_df_14['antielite_salience'] = ches_df_14['antielite_salience'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_14 = ches_df_14[['cname', 'party_name', 'antielite_salience']]
    # Rename "party_name" col
    ches_df_14.rename(columns={"party_name": "party", "cname": "country"}, inplace=True)

    # Split entries for parties with two naming conventions to separate entries
    add_entries_14 = pd.DataFrame()
    for index, row in ches_df_14.iterrows():
        # Find entries with two party namings (such as 'EDU/UDF')
        match = re.search(r'([^\/]+)\/([^\/]+)', row.party)
        if match:
            # Make first party the name of current row
            ches_df_14.loc[index, 'party'] = match.group(1)
            # Construct new row with second name and copy values for other cols
            entry = pd.DataFrame({'country': [row.country],
                                  'party': [match.group(2)],
                                  'antielite_salience': [row.antielite_salience]
                                  })
            # Add row to df which will be appended at the end
            add_entries_14 = add_entries_14.append(entry)

    # Add additional rows
    ches_df_14 = ches_df_14.append(add_entries_14)

    # Standardize party naming: lowercase, remove Umlaute, remove empty spaces, etc
    ches_df_14['party'] = ches_df_14['party'].apply(lambda x: standardize_party_naming(x))

    # Replace keys for countries
    ches_df_14['country'].replace({"ger": "de", "aus": "au", "swi": "cd"}, inplace=True)

    # Group by country
    ches_df_14_grpd = ches_df_14.groupby('country')

    dict_ches_14 = {}

    # Generate party list per country
    for index, row in ches_df_14_grpd:
        # Pop parties
        country_pop = row.loc[(row.antielite_salience == 1)]['party'].tolist()
        country_nonpop = row.loc[(row.antielite_salience == 0)]['party'].tolist()

        # Generate nested dict
        dict_ches_14[index] = {}
        dict_ches_14[index]['pop'] = country_pop
        dict_ches_14[index]['nonpop'] = country_nonpop


    # CHES 17:
    ches_df_17 = pd.read_csv(f'{data_path}\\CHES\\CHES_means_2017.csv')
    ches_df_17 = ches_df_17.loc[ches_df_17.country.isin(rel_countries)]
    ches_df_17['people_vs_elite'] = ches_df_17['people_vs_elite'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_17['antielite_salience'] = ches_df_17['antielite_salience'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_17 = ches_df_17[['country', 'party', 'people_vs_elite', 'antielite_salience']]

    # Split entries for parties with two naming conventions to separate entries
    add_entries_17 = pd.DataFrame()
    for index, row in ches_df_17.iterrows():
        # Find entries with two party namings (such as 'EDU/UDF')
        match = re.search(r'([^\/]+)\/([^\/]+)', row.party)
        if match:
            # Make first party the name of current row
            ches_df_17.loc[index, 'party'] = match.group(1)
            # Construct new row with second name and copy values for other cols
            entry = pd.DataFrame({'country': [row.country],
                                  'party': [match.group(2)],
                                  'people_vs_elite': [row.people_vs_elite],
                                  'antielite_salience': [row.antielite_salience]
                                  })
            # Add row to df which will be appended at the end
            add_entries_17 = add_entries_17.append(entry)

    # Add additional rows
    ches_df_17 = ches_df_17.append(add_entries_17)

    # Standardize party naming: lowercase, remove Umlaute, remove empty spaces, etc
    ches_df_17['party'] = ches_df_17['party'].apply(lambda x: standardize_party_naming(x))

    # Replace keys for countries
    ches_df_17['country'].replace({"ger": "de", "aus": "au", "swi": "cd"}, inplace=True)

    # Group by country
    ches_df_17_grpd = ches_df_17.groupby('country')

    dict_ches_17 = {}

    # Generate party list per country
    for index, row in ches_df_17_grpd:
        # Pop parties
        country_pop = row.loc[(row.people_vs_elite == 1) | (row.antielite_salience == 1)]['party'].tolist()
        country_nonpop = row.loc[(row.people_vs_elite == 0) & (row.antielite_salience == 0)]['party'].tolist()

        # Generate nested dict
        dict_ches_17[index] = {}
        dict_ches_17[index]['pop'] = country_pop
        dict_ches_17[index]['nonpop'] = country_nonpop

    # CHES 19:
    ches_df_19 = pd.read_csv(f'{data_path}\\CHES\\CHES2019V3.csv')
    ches_df_19 = ches_df_19.loc[ches_df_19.country.isin(rel_country_ids)]
    ches_df_19['people_vs_elite'] = ches_df_19['people_vs_elite'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_19['antielite_salience'] = ches_df_19['antielite_salience'].apply(lambda x: 1 if x > 5 else 0)
    ches_df_19 = ches_df_19[['country', 'party', 'people_vs_elite', 'antielite_salience']]

    # Split entries for parties with two naming conventions to separate entries
    add_entries_19 = pd.DataFrame()
    for index, row in ches_df_19.iterrows():
        # Find entries with two party namings (such as 'EDU/UDF')
        match = re.search(r'([^\/]+)\/([^\/]+)', row.party)
        if match:
            # Make first party the name of current row
            ches_df_19.loc[index, 'party'] = match.group(1)
            # Construct new row with second name and copy values for other cols
            entry = pd.DataFrame({'country': [row.country],
                                  'party': [match.group(2)],
                                  'people_vs_elite': [row.people_vs_elite],
                                  'antielite_salience': [row.antielite_salience]
                                  })
            # Add row to df which will be appended at the end
            add_entries_19 = add_entries_19.append(entry)

    # Add additional rows
    ches_df_19 = ches_df_19.append(add_entries_19)

    # Standardize party naming: lowercase, remove Umlaute, remove empty spaces, etc
    ches_df_19['party'] = ches_df_19['party'].apply(lambda x: standardize_party_naming(x))

    # Replace country_ids with country names
    ches_df_19['country'].replace({3: "de", 13: "au", 36: "cd"}, inplace=True)

    # Group by country
    ches_df_19_grpd = ches_df_19.groupby('country')

    dict_ches_19 = {}

    # Generate party list per country
    for index, row in ches_df_19_grpd:
        # Pop parties
        country_pop = row.loc[(row.people_vs_elite == 1) | (row.antielite_salience == 1)]['party'].tolist()
        country_nonpop = row.loc[(row.people_vs_elite == 0) & (row.antielite_salience == 0)]['party'].tolist()

        # Generate nested dict
        dict_ches_19[index] = {}
        dict_ches_19[index]['pop'] = country_pop
        dict_ches_19[index]['nonpop'] = country_nonpop

    return dict_ches_14, dict_ches_17, dict_ches_19

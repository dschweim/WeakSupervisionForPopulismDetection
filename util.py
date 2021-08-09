import os
import numpy as np
import re
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def generate_train_dev_test_split(df):
    """
    Generate train dev test split
    :param df: Dataset to split
    :type df: DataFrame
    :return: Returns three datasets
    :rtype: DataFrame, DataFrame, DataFrame
    """

    # Split into 60% train, 20% dev and 20% test
    rest, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df.POPULIST)
    train, dev = train_test_split(rest, test_size=0.1, random_state=42, shuffle=True, stratify=rest.POPULIST)

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

        if re.search(r'gruene', party):
            party = 'grune'

        if re.search(r'cdu', party):
            party = 'cdu/csu'

        if re.search(r'csu', party):
            party = 'cdu/csu'

        return party


def retrieve_year(date: str):
    """
    Retrieve year from date
    :param date: String of date
    :type date: str
    :return: Returns year of date
    :rtype:  str
    """

    # Retrieve year from date column
    year = re.search(r'^\d\d.\d\d.(\d{4})', date)

    if year is None:
        return None
    else:
        return year.group(1)


def extract_parsed_lemmas(segment):
    """
    Retrieve lemma, pos tag, dependence tag, and corresponding head token for each token in segment
    :param segment: parsed segment
    :type segment: spacy.tokens.doc.Doc
    :return: list of tokens and their parsed info
    :rtype:  list
    """

    # Get HEAD tokens
    heads = [token for token in segment if token.head == token]

    # Define list for lemmas and tags
    lemma_list = []

    # Iterate over heads
    for head in heads:

        lemmas = []
        current_sent = head.sent

        # Iterate over tokens per sentence
        for token in current_sent:
            lemmas.append((token.lemma_.lower(), token.pos_, token.dep_, token.head.text))

        lemma_list.append(lemmas)
    return lemma_list


def extract_dep_tuples(segment, id):
    """
    Retrieve tuples dict of type (subj, verb, verb_prefix, object, negation) for each verb in Segment
    :param segment: parsed segment
    :type segment: spacy.tokens.doc.Doc
    :return: list of triple dicts
    :rtype:  list
    """

    # Define individual components
    VERBS = ['VERB', 'AUX']
    VERBCOMPONENTS = ['svp']
    SUBJECTS = ['sb', 'sbp']
    OBJECTS = ['oa', 'og', 'da', 'pd']  # oc,?
    NEGATIONS = ['ng']

    # Generate empty dict list
    triples_dict_list = []

    # Extract verb & aux tokens
    verbs = [token for token in segment if token.pos_ in VERBS]

    # Iterate over verbs
    for verb in verbs:

        # # only for debugging
        lemmas = []  ##
        current_sent = verb.sent  ##
        for token in current_sent:  ##
            lemmas.append((token.lemma_.lower(), token.pos_, token.dep_, token.head.text))  ##

        # Create empty list for components
        verb_list = []
        verb_comp_list = []
        subj_list = []
        obj_list = []
        neg_list = []

        # Extract current verb
        verb_list.append(verb.lemma_.lower())

        # Iterate over verb dependents
        for child in verb.children:

            # Extract separable verb prefix
            if child.dep_ in VERBCOMPONENTS:
                verb_comp_list.append(child.lemma_.lower())

            # Extract subject
            if child.dep_ in SUBJECTS:
                subj_list.append(child.lemma_.lower())

            # Extract object
            elif child.dep_ in OBJECTS:
                obj_list.append(child.lemma_.lower())

            # Extract negation
            elif child.dep_ in NEGATIONS:
                neg_list.append(child.lemma_.lower())

            # Extract oc object for tuples without object
            elif child.dep_ == 'oc':
                if not obj_list:
                    obj_list.append(child.lemma_.lower())

        # if lists are empty, return None
        if (not verb_list) & (not verb_comp_list) & (not subj_list) & (not obj_list) & (not neg_list):
            triples_dict = None

        # Else, return content
        else:
            triples_dict = {'subject': subj_list,
                            'verb': verb_list,
                            'verb_prefix': verb_comp_list,
                            'object': obj_list,
                            'negation': neg_list}

        # Generate list with dict for each verb in Segment
        triples_dict_list.append(triples_dict)

    return triples_dict_list


def get_all_svo_tuples(svo_dict_series: pd.Series, get_components: dict):
    """
    Extract all distinct svo(+verbprefix)(+neg) tuples globally in corpus and count their occurrences
    :param get_components: dictionary that indicates which components to return s, v, o, verbprefix, neg
    :type get_components: dict
    :param svo_dict_series: series of svo-triple dicts per segment
    :type svo_dict_series: pd.Series
    :return: df of requested tuples and their count
    :rtype: pd.DataFrame
    """

    # Define empty list for tuples
    corpus_tuples = pd.DataFrame()

    # Extract boolean indicator per component from dict
    get_subj = get_components['subj']
    get_verb = get_components['verb']
    get_verbprefix = get_components['verbprefix']
    get_obj = get_components['obj']
    get_neg = get_components['neg']

    # Iterate over series
    for index, value in svo_dict_series.items():

        # Iterate over dicts (i.e. number of sentences)
        for elem in value:
            # Skip None values
            if isinstance(elem, dict):
                current_val = list(elem.values())

                # Extract requested components + negation in any case
                subj = ', '.join(current_val[0])  # subject
                verb = ', '.join(current_val[1])  # verb
                verb_prefix = ', '.join(current_val[2])  # verb_prefix
                obj = ', '.join(current_val[3])  # object
                neg = ', '.join(current_val[4])  # negation

                # Generate requested tuple and append to global list
                if get_subj & get_verb & get_verbprefix & get_obj & get_neg:
                    current_tuple = (subj, verb, verb_prefix, obj, neg)  # svo + prefix + neg
                elif get_subj & get_verb & get_verbprefix & get_obj & (not get_neg):  # svo + prefix
                    current_tuple = (subj, verb, verb_prefix, obj)

                elif get_subj & get_verb & (not get_verbprefix) & get_obj & get_neg:  # svo + neg
                    current_tuple = (subj, verb, obj, neg)
                elif get_subj & get_verb & (not get_verbprefix) & get_obj & (not get_neg):  # svo
                    current_tuple = (subj, verb, obj)

                elif get_subj & get_verb & (not get_verbprefix) & (not get_obj) & get_neg:  # sv +neg
                    current_tuple = (subj, verb, neg)
                elif get_subj & get_verb & (not get_verbprefix) & (not get_obj) & (not get_neg):  # sv
                    current_tuple = (subj, verb)

                elif (not get_subj) & get_verb & (not get_verbprefix) & get_obj & get_neg:  # vo +neg
                    current_tuple = (verb, obj, neg)
                elif (not get_subj) & get_verb & (not get_verbprefix) & get_obj & (not get_neg):  # vo
                    current_tuple = (verb, obj)

                elif get_subj & (not get_verb) & (not get_verbprefix) & get_obj & (not get_neg):  # so
                    current_tuple = (subj, obj)

                elif (not get_subj) & get_verb & (not get_verbprefix) & (not get_obj) & get_neg:  # v + neg
                    current_tuple = (verb, neg)
                elif (not get_subj) & get_verb & (not get_verbprefix) & (not get_obj) & (not get_neg):  # v
                    current_tuple = (verb)

                elif get_subj & (not get_verb) & (not get_verbprefix) & (not get_obj) & (not get_neg):  # s
                    current_tuple = (subj)

                elif (not get_subj) & (not get_verb) & (not get_verbprefix) & get_obj & (not get_neg):  # o
                    current_tuple = (obj)

                else:  # not implemented
                    raise Exception('This svo combination is not supported')

                # ignore tuples with empty components
                if all(current_tuple):
                    tuple = pd.DataFrame({'tuple': [current_tuple],
                                          'source': [index]})
                    corpus_tuples = corpus_tuples.append(tuple)

    # Generate df
    if not corpus_tuples.empty:
        tuples_df = corpus_tuples.groupby('tuple', as_index=False).agg({'tuple': ['first', 'count'],
                                                                        'source': lambda x: list(x)})

        tuples_df.columns = tuples_df.columns.droplevel(0)
        tuples_df.rename(columns={tuples_df.columns[0]: 'tuple',
                                  tuples_df.columns[1]: 'count',
                                  tuples_df.columns[2]: 'source'}, inplace=True)

    else:
        tuples_df = pd.DataFrame({'tuple': [None],
                                  'count': [None],
                                  'source': [None]})

    return tuples_df


def output_and_store_endmodel_results(output_path, classifier, feature, Y_test, Y_pred, X_test, hyperparameters):
    """
    Print results in console and store them in csv (merging with previous results)
    :param output_path: path to data output
    :type output_path: str
    :param classifier: model used in current run
    :type classifier: str
    :param feature: vectorization used in current run
    :type feature: str
    :param Y_test: ground truth labels of test set
    :type Y_test: pd.Series
    :param Y_pred: predicted labels of test set
    :type Y_pred: np.ndarray
    :param X_test: test set
    :type Y_pred: pd.DataFrame
    :param hyperparameters: tuned hyperparameters retrieved from model
    :type hyperparameters: dict
    :return:
    :rtype:
    """

    print('---------------------------------------')
    print(f"Model: {classifier}, Feature: {feature}")
    print(f"Model Test Accuracy: {accuracy_score(Y_test, Y_pred)}")
    print(f"Model Test Precision: {precision_score(Y_test, Y_pred)}")
    print(f"Model Test Recall: {recall_score(Y_test, Y_pred)}")
    print(f"Model Test F1: {f1_score(Y_test, Y_pred, average='binary')}")

    # Save results
    timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

    results_df = pd.DataFrame({'model': [classifier],
                               'vectorization': [feature],
                               'hyperparameters': [hyperparameters],
                               'accuracy': [accuracy_score(Y_test, Y_pred)],
                               'precision': [precision_score(Y_test, Y_pred)],
                               'recall': [recall_score(Y_test, Y_pred)],
                               'f1': [f1_score(Y_test, Y_pred, average='binary')],
                               'timestamp': [timestamp]
                               })
    results_df = results_df.set_index(['model', 'vectorization'])

    # If results file exists, append results to file
    if os.path.isfile(f'{output_path}\\Results\\results.csv'):
        prev_results = pd.read_csv(f'{output_path}\\Results\\results.csv',
                                   index_col=['model', 'vectorization'])

        results_df = results_df.append(prev_results)

        # only keep newest run
        results_df = results_df[~results_df.index.duplicated()].sort_index()

    results_df.to_csv(f'{output_path}\\Results\\results.csv')

    # Save individual predictions and corresponding information
    model_preds = pd.DataFrame({'Content': X_test.content,
                                'POPULIST_PeopleCent': X_test.POPULIST_PeopleCent,
                                'POPULIST_AntiElite': X_test.POPULIST_AntiElite,
                                'POPULIST_Sovereign': X_test.POPULIST_Sovereign,
                                'Country': X_test.Sample_Country,
                                'Category': X_test.Sample_Type,
                                'Y_test': Y_test.astype(int),
                                'Y_pred': Y_pred})

    model_preds.to_csv(f'{output_path}\\Results\\{classifier}_{feature}_preds.csv')

from sklearn.model_selection import train_test_split
import numpy as np
import re
import pandas as pd
from collections import Counter


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


def extract_dep_triples(segment):
    """
    Retrieve svo-triple dict of type (subj, verb, verb_prefix, object, negation) for each verb in Segment
    :param segment: parsed segment
    :type segment: spacy.tokens.doc.Doc
    :return: list of triple dicts
    :rtype:  list
    """

    # Define individual components
    VERBCOMPONENTS = ['svp']
    SUBJECTS = ['sb', 'sbp']
    OBJECTS = ['oa', 'og', 'da', 'pd'] # oc,?
    NEGATIONS = ['ng']

    # Generate empty dict list
    triples_dict_list = []

    # Extract fullverb tokens
    verbs = [token for token in segment if token.pos_ in ['VERB', 'AUX']]

    # Iterate over fullverbs
    for verb in verbs:

        # todo: only to debug
        lemmas = [] ##
        current_sent = verb.sent  ##
        for token in current_sent:  ##
            lemmas.append((token.lemma_.lower(), token.pos_, token.dep_, token.head.text))  ##

        # Create empty list for components
        verb_list = []
        verb_comp_list = []
        subj_list = []
        obj_list = []
        neg_list = []

        # Extract current fullverb
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


def get_all_svo_quadruples(svo_dict_series: pd.Series):
    """
    Extract all distinct svo(+verbprefix) quadruples globally in corpus and count their occurrences
    :param svo_dict_series: series of svo-triple dicts per segment
    :type svo_dict_series: pd.Series
    :return: df of svo-quadruples and their count
    :rtype: pd.DataFrame
    """

    corpus_triples = []

    for index, value in svo_dict_series.items():

        # Iterate over dicts (i.e. number of sentences)
        for elem in value:
            # Skip None values
            if isinstance(elem, dict):
                current_val = list(elem.values())

                # Extract components
                subj = ', '.join(current_val[0]) # subject
                verb = ', '.join(current_val[1]) # verb
                verb_prefix = ', '.join(current_val[2])  # verb_prefix
                obj = ', '.join(current_val[3])  # object
                neg = ', '.join(current_val[4])  # negation

                # Generate triple and append to global list
                current_tuple = (subj, verb, verb_prefix, obj, neg)
                corpus_triples.append(current_tuple)

    # Generate df
    tuples_df = pd.DataFrame({'tuple': Counter(corpus_triples).keys(),  # get unique values of quadruples
                              'count': Counter(corpus_triples).values()})  # get the elements' frequency

    return tuples_df


def get_all_svo_triples(svo_dict_series: pd.Series):
    """
    Extract all distinct svo-triples globally in corpus and count their occurrences
    :param svo_dict_series: series of svo-triple dicts per segment
    :type svo_dict_series: pd.Series
    :return: df of svo-triples and their count
    :rtype: pd.DataFrame
    """

    corpus_triples = []

    for index, value in svo_dict_series.items():

        # Iterate over dicts (i.e. number of sentences)
        for elem in value:
            # Skip None values
            if isinstance(elem, dict):
                current_val = list(elem.values())

                # Extract components
                subj = ', '.join(current_val[0])  # subject
                verb = ', '.join(current_val[1])  # verb
                # verb_prefix = ', '.join(current_val[2])
                obj = ', '.join(current_val[3])  # object
                neg = ', '.join(current_val[4])  # negation

                # Generate triple and append to global list
                current_tuple = (subj, verb, obj, neg)
                corpus_triples.append(current_tuple)

    # Generate df
    tuples_df = pd.DataFrame({'tuple': Counter(corpus_triples).keys(),  # get unique values of triples
                              'count': Counter(corpus_triples).values()})  # get the elements' frequency

    return tuples_df


def get_all_sv_doubles(svo_dict_series: pd.Series):
    """
    Extract all distinct svo-doubles globally in corpus and count their occurrences
    :param svo_dict_series: series of svo-triple dicts per segment
    :type svo_dict_series: pd.Series
    :return: df of svo-tuples and their count
    :rtype: pd.DataFrame
    """

    corpus_triples = []

    for index, value in svo_dict_series.items():

        # Iterate over dicts (i.e. number of sentences)
        for elem in value:
            # Skip None values
            if isinstance(elem, dict):
                current_val = list(elem.values())

                # Extract components
                subj = ', '.join(current_val[0])  # subject
                verb = ', '.join(current_val[1])  # verb
                # verb_prefix = ', '.join(current_val[2])
                # obj = ', '.join(current_val[3])
                neg = ', '.join(current_val[4])  # negation

                # Generate triple and append to global list
                current_tuple = (subj, verb, neg)
                corpus_triples.append(current_tuple)

    # Generate df
    tuples_df = pd.DataFrame({'tuple': Counter(corpus_triples).keys(),  # get unique values of tuples
                              'count': Counter(corpus_triples).values()})  # get the elements' frequency

    return tuples_df


def get_all_vo_doubles(svo_dict_series: pd.Series):
    """
    Extract all distinct vo-doubles globally in corpus and count their occurrences
    :param svo_dict_series: series of svo-triple dicts per segment
    :type svo_dict_series: pd.Series
    :return: df of svo-tuples and their count
    :rtype: pd.DataFrame
    """

    corpus_triples = []

    for index, value in svo_dict_series.items():

        # Iterate over dicts (i.e. number of sentences)
        for elem in value:
            # Skip None values
            if isinstance(elem, dict):
                current_val = list(elem.values())

                # Extract components
                # subj = ', '.join(current_val[0])  # subject
                verb = ', '.join(current_val[1])  # verb
                # verb_prefix = ', '.join(current_val[2])
                obj = ', '.join(current_val[3])
                neg = ', '.join(current_val[4])  # negation

                # Generate triple and append to global list
                current_tuple = (verb, obj, neg)
                corpus_triples.append(current_tuple)

    # Generate df
    tuples_df = pd.DataFrame({'tuple': Counter(corpus_triples).keys(),  # get unique values of tuples
                              'count': Counter(corpus_triples).values()})  # get the elements' frequency

    return tuples_df


def get_all_so_doubles(svo_dict_series: pd.Series):
    """
    Extract all distinct so-doubles globally in corpus and count their occurrences
    :param svo_dict_series: series of svo-triple dicts per segment
    :type svo_dict_series: pd.Series
    :return: df of svo-tuples and their count
    :rtype: pd.DataFrame
    """

    corpus_triples = []

    for index, value in svo_dict_series.items():

        # Iterate over dicts (i.e. number of sentences)
        for elem in value:
            # Skip None values
            if isinstance(elem, dict):
                current_val = list(elem.values())

                # Extract components
                subj = ', '.join(current_val[0])  # subject
                # verb = ', '.join(current_val[1])  # verb
                # verb_prefix = ', '.join(current_val[2])
                obj = ', '.join(current_val[3])
                neg = ', '.join(current_val[4])  # negation

                # Generate triple and append to global list
                current_tuple = (subj, obj, neg)
                corpus_triples.append(current_tuple)

    # Generate df
    tuples_df = pd.DataFrame({'tuple': Counter(corpus_triples).keys(),  # get unique values of tuples
                              'count': Counter(corpus_triples).values()})  # get the elements' frequency

    return tuples_df


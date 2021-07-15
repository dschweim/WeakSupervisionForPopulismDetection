from spacy.matcher import DependencyMatcher
import spacy


def extract_dep_tuples(segment):
    """
    Retrieve tuples dict of type (verb, verb_prefix, negation, subject, object) for each verb in Segment
    :param segment: parsed segment
    :type segment: spacy.tokens.doc.Doc
    :return: list of triple dicts
    :rtype:  list
    """

    # Define individual components
    VERBS = ['VERB', 'AUX']
    VERBCOMPONENTS = ['svp']
    NEGATIONS = ['ng']
    SUBJECTS = ['sb', 'sbp']
    OBJECTS = ['oa', 'og', 'da', 'pd']  # oc,?


    # Match patterns with help of spacy dependency matcher
    # Define dependency matcher
    spacy_model = 'de_core_news_lg'
    nlp_full = spacy.load(spacy_model)
    dep_matcher = DependencyMatcher(vocab=nlp_full.vocab)


    dep_pattern = [
        {
            'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS':  {'IN': VERBS}}
        },
        {
            'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
            'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'DEP': {'IN': SUBJECTS}}
        },
        {
            'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
            'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'DEP': {'IN': OBJECTS}}
        }
    ]

    dep_matcher.add('svo_verb', patterns=[dep_pattern])

    dep_matches = dep_matcher(segment)

    # if any matches found
    if dep_matches:

        for match in dep_matches:
            pattern_name = match[0]

            matches = match[1]

            verb, verb_prefix, subject, object = matches[0], matches[1], matches[2]

            # Generate empty dict list
            triples_dict_list = []

            # Else, return content

            triples_dict = {'verb': verb,
                            'subject': subject,
                            'object': object}


            # Generate list with dict for each verb in Segment
            triples_dict_list.append(triples_dict)

        return triples_dict_list

    else:
        return None



# Define verb as anchor pattern
# dep_pattern = [
#     {
#         'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': 'VERB'},
#         'LEMMA': {'IN': ['haben']}
#     },
#     {
#         'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
#         'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['Bundesregierung']}, 'DEP': 'sb'}}
# ]
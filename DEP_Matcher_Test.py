from spacy.matcher import DependencyMatcher
import spacy


def test_dep_matcher(x):
    spacy_model = 'de_core_news_lg'

    # Match patterns with help of spacy dependency matcher
    # Define dependency matcher
    nlp_full = spacy.load(spacy_model)
    dep_matcher = DependencyMatcher(vocab=nlp_full.vocab)

    # Define verb as anchor pattern
    dep_pattern = [
        {
            'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': 'VERB'},
            'LEMMA': {'IN': ['haben']}
        },
        {
            'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
            'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['Bundesregierung']}, 'DEP': 'sb'}}
    ]

    dep_matcher.add('svo_verb', patterns=[dep_pattern])

    dep_matches = dep_matcher(x)

    if dep_matches:
        print(x)
        print('test ok')

import pandas as pd
from spacy.matcher import DependencyMatcher
import spacy
import numpy as np


class Dep_Preprocessor:
    def __init__(
            self,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            spacy_model: str
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.spacy_model = spacy_model

    def setup_dep_matcher(self):
        # todo: Pre-process corpora with Dependency Matcher
        nlp_full = spacy.load(self.spacy_model)
        self.train_data['doc'] = list(nlp_full.pipe(self.train_data['text']))
        self.test_data['doc'] = list(nlp_full.pipe(self.test_data['text']))

        # Match patterns with help of spacy dependency matcher
        # Define dependency matcher
        dep_matcher = DependencyMatcher(vocab=nlp_full.vocab)

        VERBS = ['VERB', 'AUX']
        SUBJECTS = ['sb', 'sbp']
        # Define verb as anchor pattern
        pop_patterns_sv = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['haben']}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['bundesregierung']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['können']}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['ich']},
                                                           'DEP': {'IN': SUBJECTS}}
                }

            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['können']}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['man']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['haben']}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['regierung']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['haben']}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['grüne']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ]
        ]

        dep_matcher.add('sv', patterns=pop_patterns_sv)

        self.train_data['sv_pop'] = self.test_dep_matcher(self.train_data, dep_matcher)  # todo: Get name of pattern
        self.test_data['sv_pop'] = self.test_dep_matcher(self.test_data, dep_matcher)
        # train_data['doc'].apply(lambda x: test_dep_matcher(x, dep_matcher))

        # Drop doc col
        self.train_data.drop(columns=['doc'], inplace=True)
        self.test_data.drop(columns=['doc'], inplace=True)

    @staticmethod
    def test_dep_matcher(df, dep_matcher):
        array_res = np.zeros(len(df))

        df.reset_index(inplace=True)

        # todo: dep matcher- do not cosnider uppercase (Case non sensitive)!! -> is LEMMA case insensitive?
        for index, row in df.iterrows():
            # Find matches in current corpus
            dep_matches = dep_matcher(row.doc)

            # Set current row indicator True if any match, else set False
            array_res[index] = 1 if dep_matches else 0

        # Restore index
        df.set_index('index', inplace=True)

        return array_res

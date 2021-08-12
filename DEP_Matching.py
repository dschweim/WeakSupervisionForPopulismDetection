from spacy.matcher import DependencyMatcher


class DEP_Matcher:
    def __init__(
            self,
            vocab: str
    ):
        """
        Class to create the Dependency Matcher
        :param vocab: used trained Spacy pipeline
        :type: str
        """

        self.vocab = vocab

    def add_patterns(self):
        # Match patterns with help of spacy dependency matcher
        # Define dependency matcher
        dep_matcher = DependencyMatcher(vocab=self.vocab)

        VERBS = ['VERB', 'AUX']
        VERBCOMPONENTS = ['svp']
        SUBJECTS = ['sb', 'sbp']
        OBJECTS = ['oa', 'og', 'da', 'pd', 'oc']
        NEGATIONS = ['ng']

        # DEFINE POP PATTERNS (Verb as anchor pattern)
        pop_patterns_sv = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['haben']}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['bundesregierung']},
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
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['wir']},
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
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['regierung']},
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
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['man']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['wollen']}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['wir']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['stehen']}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['wir']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ]
        ]

        pop_patterns_vo = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': 'stellen'
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'LEMMA': 'sich',
                                                          'DEP': {'IN': OBJECTS}}
                }
            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': 'können'
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'LEMMA': 'werden',
                                                          'DEP': {'IN': OBJECTS}}
                }
            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': 'sein'
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'LEMMA': 'bleiben',
                                                          'DEP': {'IN': OBJECTS}}
                }
            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': 'werden'
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'LEMMA': 'umsetzen',
                                                          'DEP': {'IN': OBJECTS}}
                }
            ]
        ]

        pop_patterns_v = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {
                        'IN': ['leben', 'schützen', 'erhöhen', 'richten', 'erhalten', 'sollen',  'verkaufen', 'lässt',
                               'erfüllen', 'bezeichnen', 'kosten',    'bestätigen', 'sein', 'heißen', 'umsetzen']
                    }
                }
            ]
        ]

        pop_patterns_s = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['bundesregierung', 'regierung', 'politiker',
                                                                            'die', 'npd', 'linke', 'bürger', 'menschen',
                                                                            'unternehmen','millionen','staat','welche',
                                                                            'wir', 'von']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ]

        ]

        pop_patterns_o = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'object',
                    'RIGHT_ATTRS': {'LEMMA': {'IN': ['staat', 'möglichkeit', 'tragen', 'schaden', 'politik',  'leben',
                                                     'möglich']},
                                    'DEP': {'IN': OBJECTS}}
                }
            ]
        ]

        # DEFINE NONPOP PATTERNS (Verb as anchor pattern)
        nonpop_patterns_sv = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': 'haben'
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['spö']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ],
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': 'haben'
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['övp']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ]
        ]

        nonpop_patterns_vneg = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['können']}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'negation', 'RIGHT_ATTRS': {'LEMMA': 'nicht',
                                                            'DEP': {'IN': NEGATIONS}}
                }
            ]
        ]

        nonpop_patterns_v = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                    'LEMMA': {'IN': ['muss', 'halten', 'habe', 'zeigen', 'leisten', 'einsetzen',  'haben', 'sprechen']}
                }
            ]
        ]

        nonpop_patterns_s = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LOWER': {'IN': ['spö', 'övp', 'spd', 'spindelegger', 'er',
                                                                            'österreich', 'faymann', 'ich', 'was']},
                                                           'DEP': {'IN': SUBJECTS}}
                }
            ]

        ]

        nonpop_patterns_o = [
            [
                {
                    'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}}
                },
                {
                    'LEFT_ID': 'anchor_verb', 'REL_OP': '>',
                    'RIGHT_ID': 'object',
                    'RIGHT_ATTRS': {'LEMMA': {'IN': ['geben']},
                                    'DEP': {'IN': OBJECTS}}
                }
            ]
        ]

        dep_matcher.add('pop_sv', patterns=pop_patterns_sv)
        dep_matcher.add('pop_vo', patterns=pop_patterns_vo)
        dep_matcher.add('pop_v', patterns=pop_patterns_v)
        dep_matcher.add('pop_s', patterns=pop_patterns_s)
        dep_matcher.add('pop_o', patterns=pop_patterns_o)

        dep_matcher.add('nonpop_sv', patterns=nonpop_patterns_sv)
        dep_matcher.add('nonpop_vneg', patterns=nonpop_patterns_vneg)
        dep_matcher.add('nonpop_v', patterns=nonpop_patterns_v)
        dep_matcher.add('nonpop_s', patterns=nonpop_patterns_s)
        dep_matcher.add('nonpop_o', patterns=nonpop_patterns_o)

        return dep_matcher


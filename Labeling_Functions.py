import spacy
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
from snorkel.preprocess.nlp import SpacyPreprocessor
from spacy.language import Language
from spacy.matcher import DependencyMatcher

# Define constants
ABSTAIN = -1
NONPOP = 0
POP = 1


def get_lfs(lf_input: dict, lf_input_ches: dict, spacy_model: str):
    """
    Generate dataframe for ncr sentiment analysis
    :param lf_input: Dictionary with input for dictionary-based LFs
    :type lf_input:  Dict
    :param lf_input_ches: Dictionary with input for ches-based LFs
    :type lf_input:  Dict
    :param spacy_model: used trained Spacy pipeline
    :type: str
    :return: List of labeling funtions
    :rtype:  List
    """

    ## Preprocessors
    # spacy_preprocessor = SpacyPreprocessor(text_field="content", doc_field="doc", language=spacy_model, memoize=True)

    # Create custom component that converts lemmas to lower case
    @Language.component('lower_case_lemmas')
    def lower_case_lemmas(doc):
        for token in doc:
            token.lemma_ = token.lemma_.lower()
        return doc

    # Add custom component to pipe
    nlp_label = spacy.load(spacy_model, exclude=['ner', 'attribute_ruler'])
    nlp_label.add_pipe("lower_case_lemmas", last=True)

    # Define custom spacy preprocessor
    @preprocessor(memoize=True)
    def spacy_preprocessor(x):
        x.doc = nlp_label(x.content)
        return x

    ## Labeling Functions
    # a) Dictionary-based labeling

    # Define Schwarzbözl dict
    keywords_schwarzbozl = ["altparteien", "anpassung", "iwf",
                            "politiker", "austerität", "offenbar",
                            "neoliberal", "briten", "oligarchen",
                            "steinbrück", "darlehen", "steuergeld",
                            "venezuela", "dschihadist", "steuerverschwendung",
                            "bip", "entzogen", "erwerbslose",
                            "lobby", "etabliert", "souveränität",
                            "parlament", "fahrt", "rundfunkbeitrag",
                            "verlangt", "konzern", "leistet",
                            "verlust", "herhalten", "rente"]

    # Define Rooduijn dict
    keywords_rooduijn = ["elit", "konsens", "undemokratisch", "referend", "korrupt",
                         "propagand", "politiker", "täusch", "betrüg", "betrug",
                         "verrat", "scham", "schäm", "skandal", "wahrheit", "unfair",
                         "unehrlich", "establishm", "herrsch", "lüge"]

    # LF based on Schwarzbözl keywords
    @labeling_function()
    def lf_keywords_schwarzbozl(x):
        # Return a label of POP if keyword in content, otherwise ABSTAIN
        return POP if any(keyword in x.content.lower() for keyword in keywords_schwarzbozl) else ABSTAIN

    # LF based on Schwarzbözl keywords lemma
    nlp = spacy.load(spacy_model)
    lemmas_schwarzbozl = list(nlp.pipe(keywords_schwarzbozl))

    for i in range(len(lemmas_schwarzbozl)):
        lemmas_schwarzbozl[i] = lemmas_schwarzbozl[i].doc[0].lemma_

    @labeling_function(pre=[spacy_preprocessor])
    def lf_lemma_schwarzbozl(x):
        lemmas_doc = []  # Concatenate lemmas per doc
        for token in x.doc:
            lemmas_doc.append(token.lemma_)
        if any(lemma in lemmas_doc for lemma in lemmas_schwarzbozl):
            return POP
        else:
            return ABSTAIN

    # LF based on Roodujin keywords
    @labeling_function()
    def lf_keywords_rooduijn(x):
        # Return a label of POP if keyword in content, otherwise ABSTAIN
        return POP if any(keyword in x.content.lower() for keyword in keywords_rooduijn) else ABSTAIN

    # LF based on Roodujin keywords lemma
    lemmas_roodujin = list(nlp.pipe(keywords_rooduijn))

    for i in range(len(lemmas_roodujin)):
        lemmas_roodujin[i] = lemmas_roodujin[i].doc[0].lemma_

    @labeling_function(pre=[spacy_preprocessor])
    def lf_lemma_rooduijn(x):
        lemmas_doc = []  # Concatenate lemmas per doc
        for token in x.doc:
            lemmas_doc.append(token.lemma_)
        # lemmas_doc = [x.lower() for x in lemmas_doc]
        if any(lemma in lemmas_doc for lemma in lemmas_roodujin):
            return POP
        else:
            return ABSTAIN

    # LF based on NCCR-constructed keywords
    @labeling_function()
    def lf_keywords_nccr_tfidf(x):
        keywords_nccr_tfidf = lf_input['tfidf_keywords']

        # Return a label of POP if keyword in content, otherwise ABSTAIN
        return POP if any(keyword in x.content.lower() for keyword in keywords_nccr_tfidf) else ABSTAIN

    # LF based on NCCR global-constructed keywords
    @labeling_function()
    def lf_keywords_nccr_tfidf_glob(x):
        keywords_nccr_tfidf_glob = lf_input['tfidf_keywords_global']

        # Return a label of POP if keyword in content, otherwise ABSTAIN
        return POP if any(keyword in x.content.lower() for keyword in keywords_nccr_tfidf_glob) else ABSTAIN

    # LF based on NCCR country-constructed keywords
    @labeling_function()
    def lf_keywords_nccr_tfidf_country(x):
        if x.Sample_Country == 'au':
            return POP if any(keyword in x.content.lower() for keyword in lf_input['tfidf_keywords_at']) else ABSTAIN

        elif x.Sample_Country == 'cd':
            return POP if any(keyword in x.content.lower() for keyword in lf_input['tfidf_keywords_ch']) else ABSTAIN

        elif x.Sample_Country == 'de':
            return POP if any(keyword in x.content.lower() for keyword in lf_input['tfidf_keywords_de']) else ABSTAIN

    # LF based on NCCR global-constructed keywords (chi2)
    @labeling_function()
    def lf_keywords_nccr_chi2_glob(x):
        keywords_nccr_chi2_glob = lf_input['chi2_keywords_global']

        # Return a label of POP if keyword in content, otherwise ABSTAIN
        return POP if any(keyword in x.content.lower() for keyword in keywords_nccr_chi2_glob) else ABSTAIN

    # LF based on NCCR country-constructed keywords (chi2)
    @labeling_function()
    def lf_keywords_nccr_chi2_country(x):
        if x.Sample_Country == 'au':
            return POP if any(keyword in x.content.lower() for keyword in lf_input['chi2_keywords_at']) else ABSTAIN

        elif x.Sample_Country == 'cd':
            return POP if any(keyword in x.content.lower() for keyword in lf_input['chi2_keywords_ch']) else ABSTAIN

        elif x.Sample_Country == 'de':
            return POP if any(keyword in x.content.lower() for keyword in lf_input['chi2_keywords_de']) else ABSTAIN

    # b) External Knowledge-based Labeling
    # LF based on party position estimated in CHES
    ches_14 = lf_input_ches['ches_14']
    ches_17 = lf_input_ches['ches_17']
    ches_19 = lf_input_ches['ches_19']

    @labeling_function()
    def lf_party_position_ches(x):
        if x.year < 2014:
            return ABSTAIN

        elif 2014 <= x.year < 2017:
            if x.party in ches_14[x.Sample_Country]['pop']:
                return POP
            elif x.party in ches_14[x.Sample_Country]['nonpop']:
                return NONPOP
            else:
                return ABSTAIN

        elif 2017 <= x.year < 2019:
            if x.party in ches_17[x.Sample_Country]['pop']:
                return POP
            elif x.party in ches_17[x.Sample_Country]['nonpop']:
                return NONPOP
            else:
                return ABSTAIN

        elif 2019 <= x.year:
            if x.party in ches_19[x.Sample_Country]['pop']:
                return POP
            elif x.party in ches_19[x.Sample_Country]['nonpop']:
                return NONPOP
            else:
                return ABSTAIN

    # c) DEP-based Labeling:
    # Match patterns with help of spacy dependency matcher
    # Define dependency matcher
    dep_matcher = DependencyMatcher(vocab=nlp_label.vocab)

    VERBS = ['VERB', 'AUX']
    VERBCOMPONENTS = ['svp']
    SUBJECTS = ['sb', 'sbp']
    OBJECTS = ['oa', 'og', 'da', 'pd']  # oc,?
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
        ]
    ]

    pop_patterns_v = [
        [
            {
                'RIGHT_ID': 'anchor_verb', 'RIGHT_ATTRS': {'POS': {'IN': VERBS}},
                'LEMMA': {'IN': ['leben', 'schützen', 'erhöhen', 'richten', 'erhalten', 'sollen', 'verkaufen', 'lässt',
                                 'erfüllen', 'bezeichnen', 'kosten', 'bestätigen', 'sein', 'heißen', 'umsetzen']}
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
                                                                        'unternehmen', 'millionen', 'staat', 'welche',
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
                'RIGHT_ID': 'object', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['staat', 'möglichkeit', 'politik', 'möglich']},
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
                'LEMMA': {'IN': ['muss', 'halten', 'habe', 'zeigen', 'leisten', 'einsetzen', 'haben', 'sprechen']}
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

    dep_matcher.add('pop_sv', patterns=pop_patterns_sv)
    dep_matcher.add('pop_vo', patterns=pop_patterns_vo)
    dep_matcher.add('pop_v', patterns=pop_patterns_v)
    dep_matcher.add('pop_s', patterns=pop_patterns_s)
    dep_matcher.add('pop_o', patterns=pop_patterns_o)

    dep_matcher.add('nonpop_sv', patterns=nonpop_patterns_sv)
    dep_matcher.add('nonpop_vneg', patterns=nonpop_patterns_vneg)
    dep_matcher.add('nonpop_v', patterns=nonpop_patterns_v)
    dep_matcher.add('nonpop_s', patterns=nonpop_patterns_s)

    @labeling_function(pre=[spacy_preprocessor])
    def lf_dep_pop_sv(x):

        # Get matches and the names of the patterns that caused the match
        dep_matches = dep_matcher(x.doc)
        matched_patterns = [nlp_label.vocab[i[0]].text for i in dep_matches]

        # If match and pattern name equals current lfs pattern, return POP
        if dep_matches:
            return POP if any(patt == 'pop_sv' for patt in matched_patterns) else ABSTAIN
        else:
            return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def lf_dep_pop_vo(x):

        # Get matches and the names of the patterns that caused the match
        dep_matches = dep_matcher(x.doc)
        matched_patterns = [nlp_label.vocab[i[0]].text for i in dep_matches]

        # If match and pattern name equals current lfs pattern, return POP
        if dep_matches:
            return POP if any(patt == 'pop_vo' for patt in matched_patterns) else ABSTAIN
        else:
            return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def lf_dep_pop_v(x):

        # Get matches and the names of the patterns that caused the match
        dep_matches = dep_matcher(x.doc)
        matched_patterns = [nlp_label.vocab[i[0]].text for i in dep_matches]

        # If match and pattern name equals current lfs pattern, return POP
        if dep_matches:
            return POP if any(patt == 'pop_v' for patt in matched_patterns) else ABSTAIN
        else:
            return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def lf_dep_pop_s(x):

        # Get matches and the names of the patterns that caused the match
        dep_matches = dep_matcher(x.doc)
        matched_patterns = [nlp_label.vocab[i[0]].text for i in dep_matches]

        # If match and pattern name equals current lfs pattern, return POP
        if dep_matches:
            return POP if any(patt == 'pop_s' for patt in matched_patterns) else ABSTAIN
        else:
            return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def lf_dep_pop_o(x):

        # Get matches and the names of the patterns that caused the match
        dep_matches = dep_matcher(x.doc)
        matched_patterns = [nlp_label.vocab[i[0]].text for i in dep_matches]

        # If match and pattern name equals current lfs pattern, return POP
        if dep_matches:
            return POP if any(patt == 'pop_o' for patt in matched_patterns) else ABSTAIN
        else:
            return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def lf_dep_nonpop_sv(x):

        # Get matches and the names of the patterns that caused the match
        dep_matches = dep_matcher(x.doc)
        matched_patterns = [nlp_label.vocab[i[0]].text for i in dep_matches]

        # If match and pattern name equals current lfs pattern, return NONPOP
        if dep_matches:
            return NONPOP if any(patt == 'nonpop_sv' for patt in matched_patterns) else ABSTAIN
        else:
            return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def lf_dep_nonpop_vneg(x):

        # Get matches and the names of the patterns that caused the match
        dep_matches = dep_matcher(x.doc)
        matched_patterns = [nlp_label.vocab[i[0]].text for i in dep_matches]

        # If match and pattern name equals current lfs pattern, return NONPOP
        if dep_matches:
            return NONPOP if any(patt == 'nonpop_vneg' for patt in matched_patterns) else ABSTAIN
        else:
            return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def lf_dep_nonpop_v(x):

        # Get matches and the names of the patterns that caused the match
        dep_matches = dep_matcher(x.doc)
        matched_patterns = [nlp_label.vocab[i[0]].text for i in dep_matches]

        # If match and pattern name equals current lfs pattern, return NONPOP
        if dep_matches:
            return NONPOP if any(patt == 'nonpop_v' for patt in matched_patterns) else ABSTAIN
        else:
            return ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def lf_dep_nonpop_s(x):

        # Get matches and the names of the patterns that caused the match
        dep_matches = dep_matcher(x.doc)
        matched_patterns = [nlp_label.vocab[i[0]].text for i in dep_matches]

        # If match and pattern name equals current lfs pattern, return NONPOP
        if dep_matches:
            return NONPOP if any(patt == 'nonpop_s' for patt in matched_patterns) else ABSTAIN
        else:
            return ABSTAIN

    # ## Sentiment based
    # nlp = spacy.load('de_core_news_lg')
    #
    # # sentiws = spaCySentiWS(sentiws_path='C:/Users/dschw/Documents/GitHub/Thesis/Data/SentiWS_v2.0')
    # # #nlp.add_pipe('sentiws')
    # # nlp.add_pipe('spacytextblob')
    # # doc = nlp('Die Dummheit der Unterwerfung blüht in hübschen Farben.')
    # # for token in doc:
    # #     print('{}, {}, {}'.format(token.text, token._.sentiws, token.pos_))
    # #

    # from textblob_de import TextBlobDE
    # doc = 'Die Dummheit der Unterwerfung blüht in hübschen Farben. Das ist ein hässliches Auto'
    # blob = TextBlobDE(doc)
    # print(blob.tags)
    # for sentence in blob.sentences:
    #     print(sentence.sentiment.polarity)

    # Define list of lfs to use
    list_lfs = [lf_keywords_schwarzbozl,
                lf_lemma_schwarzbozl,
                lf_keywords_rooduijn,
                lf_lemma_rooduijn,
                lf_keywords_nccr_tfidf,
                lf_keywords_nccr_tfidf_glob,
                lf_keywords_nccr_tfidf_country,
                lf_keywords_nccr_chi2_glob,
                lf_keywords_nccr_chi2_country,

                lf_dep_pop_sv,
                lf_dep_pop_vo,
                lf_dep_pop_v,
                lf_dep_pop_s,
                lf_dep_pop_o,

                lf_dep_nonpop_sv,
                lf_dep_nonpop_vneg,
                lf_dep_nonpop_v,
                lf_dep_nonpop_s,

                lf_party_position_ches]

    return list_lfs

    # todo: transformation functions (e.g. https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial)

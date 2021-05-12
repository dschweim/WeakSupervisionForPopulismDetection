import re
import spacy
import torch
import numpy as np
import pandas as pd
from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.preprocess import preprocessor
from spacy_sentiws import spaCySentiWS
import Tensor2Attr

# Define constants
ABSTAIN = -1
NONPOP = 0
POP = 1


def get_lfs(lf_input: dict, lf_input_ches: dict):

    ## Preprocessors
    spacy = SpacyPreprocessor(text_field="text", doc_field="doc",
                              language="de_core_news_lg", memoize=True)

    @preprocessor(memoize=True)
    def custom_spacy_preprocessor(x):
        nlp_trf = spacy.load("de_dep_news_trf")
        nlp_trf.add_pipe('tensor2attr')
        x.doc = nlp_trf(x.text)
        return x

    ## Labeling Functions
    # a) Dictionary-based labeling

    # LF based on Schwarzbözl keywords
    @labeling_function()
    def lf_contains_keywords_schwarzbozl(x):
        keywords_schwarbozl = ["altparteien", "anpassung", "iwf",
                               "politiker", "austerität", "offenbar",
                               "neoliberal", "briten", "oligarchen",
                               "steinbrück", "darlehen", "steuergeld",
                               "venezuela", "dschihadist", "steuerverschwendung",
                               "bip", "entzogen", "erwerbslose",
                               "lobby", "etabliert", "souveränität",
                               "parlament", "fahrt", "rundfunkbeitrag",
                               "verlangt", "konzern", "leistet",
                               "verlust", "herhalten", "rente"]

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in keywords_schwarbozl) else ABSTAIN

    # LF based on Roodujin keywords
    @labeling_function()
    def lf_contains_keywords_roodujin(x):
        keywords_roodujin = ["elit", "konsens", "undemokratisch", "referend", "korrupt",
                             "propagand", "politiker", "täusch", "betrüg", "betrug",
                             "verrat", "scham", "schäm", "skandal", "wahrheit", "unfair",
                             "unehrlich", "establishm", "herrsch", "lüge"]

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in keywords_roodujin) else ABSTAIN

    # LF based on Roodujin keywords-regex search
    @labeling_function()
    def lf_contains_keywords_roodujin_regex(x):
        regex_keywords_roodujin = ["elit\S*", "konsens\S*", "undemokratisch\S*",
                                   "referend\S*", "korrupt\S*", "propagand\S*",
                                   "politiker\S*", "täusch\S*", "betrüg\S*",
                                   "betrug\S*", "\S*verrat\S*", "scham\S*", "schäm\S*",
                                   "skandal\S*", "wahrheit\S*", "unfair\S*",
                                   "unehrlich\S*", "establishm\S*", "\S*herrsch\S*",
                                   "lüge\S*"]

        regex_roodujin = '|'.join(regex_keywords_roodujin)

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if re.search(regex_roodujin, x.text, flags=re.IGNORECASE) else ABSTAIN

    # LF based on NCCR-constructed keywords
    @labeling_function()
    def lf_contains_keywords_nccr_tfidf(x):
        regex_keywords_nccr_tfidf = lf_input['tfidf_keywords']

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in regex_keywords_nccr_tfidf) else ABSTAIN

    # LF based on NCCR global-constructed keywords
    @labeling_function()
    def lf_contains_keywords_nccr_tfidf_glob(x):
        regex_keywords_nccr_tfidf_glob = lf_input['tfidf_keywords_global']

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in regex_keywords_nccr_tfidf_glob) else ABSTAIN

    # LF based on NCCR country-constructed keywords
    @labeling_function()
    def lf_contains_keywords_nccr_tfidf_country(x):
        if x.Sample_Country == 'au':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['tfidf_keywords_at']) else ABSTAIN

        elif x.Sample_Country == 'cd':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['tfidf_keywords_ch']) else ABSTAIN

        elif x.Sample_Country == 'de':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['tfidf_keywords_de']) else ABSTAIN

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

    # c) Key Message-based Labeling:

    # LFS: Key Message 1 - Discrediting the Elite
    # negative personality and personal negative attributes of a target
    @labeling_function(pre=[spacy])
    def lf_discrediting_elite(x):
        target = 'bundesregierung'
        if target in x.text.lower():
            # {"lower": target}, IS_ADJ: True and neg
            # for token in x.doc:
            #
            #     if token.head == target and token.pos_ == 'ADJ': #& is negative & refers to target
            #         print(token.text)

            return POP
        else:
            return ABSTAIN

        # 1. find target of ELITE
        # 2. Attribute refers to ELITE
        # 3. Negative attributes

    # LFS: Key Message 2- Blaming the Elite
    # identify people embedding

    @labeling_function(pre=[custom_spacy_preprocessor])
    def lf_people_detector(x):
        # Define elite keywords
        key_word_Df = pd.DataFrame({'text': ["Bundesregierung"], 'party': None, 'Sample_Country': None,
                                 'year': None, 'POPULIST': None})
        key_word = custom_spacy_preprocessor(key_word_Df.loc[0]).doc[0]

        # Calculate embedding-based similarity of key_word tokens for tokens with relevant POS tag
        relevant_tags = ['PROPN', 'NOUN', 'PRON']

        for sent in x.doc.sents:
            for token in sent:
                if token.pos_ in relevant_tags:
                    sim = x.doc[token.i].similarity(key_word)
                    if sim > 0.2:
                        return POP
                else:
                    return ABSTAIN

    ## SPACY PATTERN MATCHING FOR KEYWORDS





    # Define list of lfs to use
    list_lfs = [lf_contains_keywords_schwarzbozl, lf_contains_keywords_roodujin, lf_contains_keywords_roodujin_regex,
                lf_contains_keywords_nccr_tfidf, lf_contains_keywords_nccr_tfidf_glob,
                lf_contains_keywords_nccr_tfidf_country, lf_discrediting_elite, lf_party_position_ches]

    return list_lfs


    #todo: transformation functions (e.g. https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial)


import re
import spacy
import torch
import numpy
import pandas as pd
from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.preprocess import preprocessor
from spacy_sentiws import spaCySentiWS

# Define constants
ABSTAIN = -1
NONPOP = 0
POP = 1

test_mpd = pd.read_csv('C:/Users/dschw/Downloads/MPDataset_MPDS2020b.csv')

def get_lfs_external_inputs():
    threshold = 5

    rel_countries = ['aus', 'ger', 'swi']
    rel_country_ids = ['13', '3', '36']

    ches_df_14 = pd.read_csv('C:/Users/dschw/Documents/GitHub/Thesis/Data/CHES/2014_CHES_dataset_means.csv')
    ches_df_14 = ches_df_14.loc[ches_df_14.country.isin(rel_country_ids)]

    ches_df_17 = pd.read_csv('C:/Users/dschw/Documents/GitHub/Thesis/Data/CHES/CHES_means_2017.csv')
    ches_df_17 = ches_df_17.loc[ches_df_17.country.isin(rel_countries)]

    ches_df_19 = pd.read_csv('C:/Users/dschw/Documents/GitHub/Thesis/Data/CHES/CHES2019V3.csv')
    ches_df_19 = ches_df_19.loc[ches_df_19.country.isin(rel_country_ids)]
    ches_df_19['people_vs_elite'] = ches_df_19['people_vs_elite'].apply(lambda x: [0 if y <= 5 else 1 for y in x])
    ches_df_19['antielite_salience'] = ches_df_19['antielite_salience'].apply(lambda x: [0 if y <= 5 else 1 for y in x])
    ches_df_19 = [['party', 'people_vs_elite', 'antielite_salience']]

    ches_19 = {'pop': [],
               'nonpop': []}

    #todo: Preprocess to reduce columns & define threshold when to consider POP and when not
    ## todo: return dictionary with list of parties that are populist/nonpopulist
    # todo: make sure party name matches to format of data.party

    return ches_14, ches_17, ches_19

def get_lfs(lf_input: dict):
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
    def lf_contains_keywords_nccr_tfidf_ctry(x):
        if x.Sample_Country == 'au':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['tfidf_keywords_at']) else ABSTAIN

        elif x.Sample_Country == 'cd':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['tfidf_keywords_ch']) else ABSTAIN

        elif x.Sample_Country == 'de':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['tfidf_keywords_de']) else ABSTAIN

    # b) External Knowledge-based Labeling
    ches_14, ches_17, ches_19 = get_lfs_external_inputs()

    @labeling_function()
    def lf_party_position_ches(x):
        if x.year < 2014:
            return ABSTAIN

        elif 2014 <= x.year < 2017:
            if x.party in ches_14['pop']:
                return POP
            elif x.party in ches_14['nonpop']:
                return NONPOP
            else:
                return ABSTAIN

        elif 2017 <= x.year < 2019:
            if x.party in ches_17['pop']:
                return POP
            elif x.party in ches_17['nonpop']:
                return NONPOP
            else:
                return ABSTAIN

        elif 2019 <= x.year:
            if x.party in ches_19['pop']:
                return POP
            elif x.party in ches_19['nonpop']:
                return NONPOP
            else:
                return ABSTAIN



    # todo: c) Spacy-based labeling
    # Preprocessor for sentiment
    @preprocessor(memoize=True)
    def sentiment_preprocessor(x):

        return x

    # d) Key Message-based Labeling:
    custom_spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

    # LFS: Key Message 1 - Discrediting the Elite
    # negative personality and personal negative attributes of a target
    @labeling_function()
    def lf_discrediting_elite(x):

        target = 'bundesregierung'
        if target in x.text.lower():
            return POP
        else:
            return ABSTAIN

        # 1. find target of ELITE
        # 2. Attribute refers to ELITE
        # 3. Negative attributes

    # LFS: Key Message 2- Blaming the Elite
    # @labeling_function(pre=[spac])
    # def km2_blaming_elite(x):

    # Define list of lfs to use
    list_lfs = [lf_contains_keywords_schwarzbozl, lf_contains_keywords_roodujin, lf_contains_keywords_roodujin_regex,
                lf_contains_keywords_nccr_tfidf, lf_contains_keywords_nccr_tfidf_glob,
                lf_contains_keywords_nccr_tfidf_ctry, lf_discrediting_elite]

    return list_lfs

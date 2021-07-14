import spacy
import pandas as pd
from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.preprocess import preprocessor
import numpy as np
import Tensor2Attr
from spacy.matcher import DependencyMatcher
from util import extract_dep_tuples, get_all_svo_tuples, get_svo_tuples_segment

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
    spacy_preprocessor = SpacyPreprocessor(text_field="text", doc_field="doc", language=spacy_model, memoize=True)

    @preprocessor(memoize=True)
    def custom_spacy_preprocessor(x):
        #nlp_trf = spacy.load("de_dep_news_trf")
        #nlp_trf.add_pipe('tensor2attr')

        nlp_full = spacy.load(spacy_model)
        x.doc = nlp_full(x.text)
        x.tuples = extract_dep_tuples(x.doc)

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
        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in keywords_schwarzbozl) else ABSTAIN

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
        #lemmas_doc = [x.lower() for x in lemmas_doc]
        if any(lemma in lemmas_doc for lemma in lemmas_schwarzbozl):
            return POP
        else:
            return ABSTAIN

    # LF based on Roodujin keywords
    @labeling_function()
    def lf_keywords_rooduijn(x):
        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in keywords_rooduijn) else ABSTAIN

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

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in keywords_nccr_tfidf) else ABSTAIN

    # LF based on NCCR global-constructed keywords
    @labeling_function()
    def lf_keywords_nccr_tfidf_glob(x):
        keywords_nccr_tfidf_glob = lf_input['tfidf_keywords_global']

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in keywords_nccr_tfidf_glob) else ABSTAIN

    # LF based on NCCR country-constructed keywords
    @labeling_function()
    def lf_keywords_nccr_tfidf_country(x):
        if x.Sample_Country == 'au':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['tfidf_keywords_at']) else ABSTAIN

        elif x.Sample_Country == 'cd':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['tfidf_keywords_ch']) else ABSTAIN

        elif x.Sample_Country == 'de':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['tfidf_keywords_de']) else ABSTAIN

    # LF based on NCCR global-constructed keywords (chi2)
    @labeling_function()
    def lf_keywords_nccr_chi2_glob(x):
        keywords_nccr_chi2_glob = lf_input['chi2_keywords_global']

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in keywords_nccr_chi2_glob) else ABSTAIN

    # LF based on NCCR country-constructed keywords (chi2)
    @labeling_function()
    def lf_keywords_nccr_chi2_country(x):
        if x.Sample_Country == 'au':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['chi2_keywords_at']) else ABSTAIN

        elif x.Sample_Country == 'cd':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['chi2_keywords_ch']) else ABSTAIN

        elif x.Sample_Country == 'de':
            return POP if any(keyword in x.text.lower() for keyword in lf_input['chi2_keywords_de']) else ABSTAIN


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

    tuples_pop = \
        lf_input['chi2_dicts_pop']['{\'subj\': True, \'verb\': True, \'verbprefix\': False, \'obj\': False, \'neg\': False}']


    ## todo: Transform to array
    b = np.array([tuples_pop]).T

    #
    # @labeling_function(pre=[custom_spacy_preprocessor])
    # def lf_dep_dict_pop_svo(x):
    #
    #     # Extract tuples from x #todo: put this into preprocessor
    #     get_components = {'subj': True, 'verb': True, 'verbprefix': False, 'obj': False, 'neg': False}
    #     #segment_tuples = get_svo_tuples(x.tuples, get_components).tuple.values
    #
    #     segment_tuples = get_svo_tuples_segment(x.tuples, get_components)
    #
    #     if np.any(np.isin(tuples_pop, segment_tuples)):
    #         # print('yes')
    #         return POP
    #     else:
    #         return ABSTAIN

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
        'RIGHT_ID': 'subject', 'RIGHT_ATTRS': {'LEMMA': {'IN': ['Bundesregierung']},'DEP': 'subj'}}
    ]

    dep_matcher.add('svo_verb', patterns=[dep_pattern])

    @labeling_function(pre=[custom_spacy_preprocessor])
    def lf_dep_dict_pop_svo(x):

        # Try to find any dep match in texts
        dep_matches = dep_matcher(x.doc)

        # If length of result >0
        if dep_matches:
            print('test')
            return POP
        else:
            return NONPOP


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
                lf_party_position_ches,
                lf_dep_dict_pop_svo]

    return list_lfs

    # todo: transformation functions (e.g. https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial)

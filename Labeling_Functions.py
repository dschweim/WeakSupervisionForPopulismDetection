import re
import spacy
import torch
import numpy
from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.preprocess import preprocessor
from spacy_sentiws import spaCySentiWS

# Define constants
ABSTAIN = -1
NONPOP = 0
POP = 1


def get_lfs(lf_input: dict):
    # a) Dictionary-based labeling
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

    @labeling_function()
    def lf_contains_keywords_roodujin(x):
        keywords_roodujin = ["elit", "konsens", "undemokratisch", "referend", "korrupt",
                             "propagand", "politiker", "täusch", "betrüg", "betrug",
                             "verrat", "scham", "schäm", "skandal", "wahrheit", "unfair",
                             "unehrlich", "establishm", "herrsch", "lüge"]

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in keywords_roodujin) else ABSTAIN

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

    @labeling_function()
    def lf_contains_keywords_nccr_tfidf(x):
        regex_keywords_nccr_tfidf = lf_input['tfidf_keywords']

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in regex_keywords_nccr_tfidf) else ABSTAIN

    @labeling_function()
    def lf_contains_keywords_nccr_tfidf_glob(x):
        regex_keywords_nccr_tfidf_glob = lf_input['tfidf_keywords_global']

        # Return a label of POP if keyword in text, otherwise ABSTAIN
        return POP if any(keyword in x.text.lower() for keyword in regex_keywords_nccr_tfidf_glob) else ABSTAIN

    # todo: Include country-spec keywords

    # todo: b) Spacy-based labeling
    # Preprocessor for sentiment
    @preprocessor(memoize=True)
    def sentiment_preprocessor(x):

        return x

    # c) Key Message-based Labeling:
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
                lf_discrediting_elite]

    return list_lfs

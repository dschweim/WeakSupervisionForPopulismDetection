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

def get_lfs (lf_input: dict):

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
    # todo: include Preprocessor
    # @Preprocessor
    #
    # is_using_gpu = spacy.prefer_gpu()
    # if is_using_gpu:
    #     torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # # Install pretrained transformer
    # nlp = spacy.load("de_dep_news_trf")
    # doc = nlp("Das ist ein Text")
    #
    # apple1 = nlp("Sie belügen das Volk")
    # apple2 = nlp("Machen Sie Politik für das Volk und nicht gegen das Volk!")
    # apple3 = nlp("wir stehen dem Anliegen, das Volk direkt entscheiden zu lassen, sehr offen gegenüber")
    # print(apple1[3].similarity(apple2[3]))  # 0.73428553
    # print(apple1[0].similarity(apple3[0]))

    text = "Sie belügen das Volk"
    # # todo: spacy preprocessing
    import de_dep_news_trf
    # nlp = de_dep_news_trf.load()
    # sentiws = spaCySentiWS(sentiws_path='C:/Users/dschw/Downloads/SentiWS_v2.0')
    # nlp.add_pipe('sentiws')
    # doc = nlp(text)
    # print([(w.text, w.pos_) for w in doc])
    # for token in doc:
    #     print('{}, {}, {}'.format(token.text, token._.sentiws, token.pos_))
    # token_list = [token for token in doc]

    # Preprocessor for sentiment
    @preprocessor(memoize=True)
    def sentiment_preprocessor(x):

        return x

    from germansentiment import SentimentModel

    model = SentimentModel()

    texts = [
        "Sie belügen das Volk",
        "Machen Sie Politik für das Volk und nicht gegen das Volk!",
        "wir stehen dem Anliegen, das Volk direkt entscheiden zu lassen, sehr offen gegenüber"]

    result = model.predict_sentiment(texts)
    print(result)

    # c) Key Message-based Labeling:
    custom_spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

    # LFS: Key Message 1 - Discrediting the Elite
    # negative personality and personal negative attributes of a target
    @labeling_function(pre=sentiment_preprocessor)
    def lf_discrediting_elite(x):
        if any([ent.label_ == "WORK_OF_ART" for ent in x.doc.ents]):
            return POP
        # if model.predict_sentiment(x):
        #     return POP
        # # Return a label of POP if keyword in text, otherwise ABSTAIN
        # return POP if any(keyword in x.text.lower() for keyword in regex_keywords_nccr_tfidf_glob) else ABSTAIN


    #2sentiment == NEG?


    # LFS: Key Message 2- Blaming the Elite
    #@labeling_function(pre=[spac])
    #def km2_blaming_elite(x):


    # Define list of lfs to use
    list_lfs = [lf_contains_keywords_schwarzbozl, lf_contains_keywords_roodujin, lf_contains_keywords_roodujin_regex,
                lf_contains_keywords_nccr_tfidf, lf_contains_keywords_nccr_tfidf_glob,
                lf_discrediting_elite]

    return list_lfs
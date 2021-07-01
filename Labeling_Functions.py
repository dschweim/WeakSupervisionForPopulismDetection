import spacy
import pandas as pd
from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.preprocess import preprocessor

# Define constants
ABSTAIN = -1
NONPOP = 0
POP = 1


def get_lfs(lf_input: dict, lf_input_ches: dict):
    """
    Generate dataframe for ncr sentiment analysis
    :param lf_input: Dictionary with input for dictionary-based LFs
    :type lf_input:  Dict
    :param lf_input_ches: Dictionary with input for ches-based LFs
    :type lf_input:  Dict
    :return: List of labeling funtions
    :rtype:  List
    """
    ## Preprocessors
    de_spacy = SpacyPreprocessor(text_field="text", doc_field="doc",
                                 language="de_core_news_lg", memoize=True)

    @preprocessor(memoize=True)
    def custom_spacy_preprocessor(x):
        nlp_trf = spacy.load("de_dep_news_trf")
        nlp_trf.add_pipe('tensor2attr')
        x.doc = nlp_trf(x.text)
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
    nlp = spacy.load("de_core_news_lg")
    lemmas_schwarzbozl = list(nlp.pipe(keywords_schwarzbozl))

    for i in range(len(lemmas_schwarzbozl)):
        lemmas_schwarzbozl[i] = lemmas_schwarzbozl[i].doc[0].lemma_

    @labeling_function(pre=[de_spacy])
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

    @labeling_function(pre=[de_spacy])
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

    # c) Key Message-based Labeling:

    # LFS: Key Message 1 - Discrediting the Elite
    # negative personality and personal negative attributes of a target

    ## neg sent analysis
    #SENTIMENT IS NEG
    @labeling_function(pre=[de_spacy])
    def lf_discrediting_elite(x):
        #for chunk in x.doc.noun_chunks:
            #print(chunk.text)

        target = 'bundesregierung'
        if target in x.text.lower():
            return POP
            # {"lower": target}, IS_ADJ: True and nfeg
            # for token in x.doc:
            #
            #     if token.head == target and token.pos_ == 'ADJ': #& is negative & refers to target
            #         print(token.text)

            #return POP
        else:
            return ABSTAIN

        # 1. find target of ELITE

        # 2. Attribute refers to ELITE
        # 3. Negative attributes


    ## Anti-Elite
    #keywords_elite
    # @labeling_function(pre=[de_spacy])
    # def lf_anti_elite(x):


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




    ## Sentiment based
    nlp = spacy.load('de_core_news_lg')

    # sentiws = spaCySentiWS(sentiws_path='C:/Users/dschw/Documents/GitHub/Thesis/Data/SentiWS_v2.0')
    # #nlp.add_pipe('sentiws')
    # nlp.add_pipe('spacytextblob')
    # doc = nlp('Die Dummheit der Unterwerfung blüht in hübschen Farben.')
    # for token in doc:
    #     print('{}, {}, {}'.format(token.text, token._.sentiws, token.pos_))
    #

    from textblob_de import TextBlobDE
    doc = 'Die Dummheit der Unterwerfung blüht in hübschen Farben. Das ist ein hässliches Auto'
    blob = TextBlobDE(doc)
    print(blob.tags)
    for sentence in blob.sentences:
        print(sentence.sentiment.polarity)


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
                lf_discrediting_elite,
                lf_party_position_ches]

    return list_lfs

    # todo: transformation functions (e.g. https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial)

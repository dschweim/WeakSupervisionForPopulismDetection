import spacy
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
from snorkel.preprocess.nlp import SpacyPreprocessor
from spacy.language import Language
from DEP_Matching import DEP_Matcher

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

    ## Preprocessor
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

        # Extract all lower lemmas in doc into list
        lemmas_doc = []
        for token in x.doc:
            lemmas_doc.append(token.lemma_)
        x.doc_lemmas = lemmas_doc
        return x

    ## Labeling Functions
    # a) Literature Dict-based labeling

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

    # 1) Keywords

    # LF based on Schwarzbözl keywords
    @labeling_function()
    def lf_keywords_schwarzbozl(x):
        # Return a label of POP if keyword in content, otherwise ABSTAIN
        return POP if any(keyword in x.content.lower() for keyword in keywords_schwarzbozl) else ABSTAIN

    # LF based on Roodujin keywords
    @labeling_function()
    def lf_keywords_rooduijn(x):
        # Return a label of POP if keyword in content, otherwise ABSTAIN
        return POP if any(keyword in x.content.lower() for keyword in keywords_rooduijn) else ABSTAIN

    # 2) Keywords Lemma

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

    # b) Custom Dictionary-based labeling
    # 1) Keywords

    # LF based on NCCR-constructed keywords
    @labeling_function()
    def lf_keywords_nccr_tfidf_av(x):
        keywords_nccr_tfidf = lf_input['tfidf_keywords_av']

        # Return a label of POP if keyword in content, otherwise ABSTAIN
        return POP if any(keyword in x.content.lower() for keyword in keywords_nccr_tfidf) else ABSTAIN

    # LF based on NCCR global-constructed keywords
    @labeling_function()
    def lf_keywords_nccr_tfidf_glob(x):
        keywords_nccr_tfidf_glob = lf_input['tfidf_keywords_global']

        # Return a label of POP if keyword in content, otherwise ABSTAIN
        return POP if any(keyword in x.content.lower() for keyword in keywords_nccr_tfidf_glob) else ABSTAIN

    # LF based on NCCR country-constructed keywords (av)
    @labeling_function()
    def lf_keywords_nccr_tfidf_av_country(x):
        if x.Sample_Country == 'au':
            return POP if any(keyword in x.content.lower() for keyword in lf_input['tfidf_keywords_av_at']) else ABSTAIN

        elif x.Sample_Country == 'cd':
            return POP if any(keyword in x.content.lower() for keyword in lf_input['tfidf_keywords_av_ch']) else ABSTAIN

        elif x.Sample_Country == 'de':
            return POP if any(keyword in x.content.lower() for keyword in lf_input['tfidf_keywords_av_de']) else ABSTAIN

    # LF based on NCCR country-constructed keywords (global)
    @labeling_function()
    def lf_keywords_nccr_tfidf_global_country(x):
        if x.Sample_Country == 'au':
            return POP if any(
                keyword in x.content.lower() for keyword in lf_input['tfidf_keywords_global_at']) else ABSTAIN

        elif x.Sample_Country == 'cd':
            return POP if any(
                keyword in x.content.lower() for keyword in lf_input['tfidf_keywords_global_ch']) else ABSTAIN

        elif x.Sample_Country == 'de':
            return POP if any(
                keyword in x.content.lower() for keyword in lf_input['tfidf_keywords_global_de']) else ABSTAIN

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

    # 2) Keywords Lemma
    # Prepare Keyword lists
    lemmas_nccr_tfidf = list(nlp.pipe(lf_input['tfidf_keywords_av']))
    for i in range(len(lemmas_nccr_tfidf)):
        lemmas_nccr_tfidf[i] = lemmas_nccr_tfidf[i].doc[0].lemma_

    lemmas_nccr_tfidf_glob = list(nlp.pipe(lf_input['tfidf_keywords_global']))
    for i in range(len(lemmas_nccr_tfidf_glob)):
        lemmas_nccr_tfidf_glob[i] = lemmas_nccr_tfidf_glob[i].doc[0].lemma_

    lemmas_nccr_tfidf_av_at = list(nlp.pipe(lf_input['tfidf_keywords_av_at']))
    for i in range(len(lemmas_nccr_tfidf_av_at)):
        lemmas_nccr_tfidf_av_at[i] = lemmas_nccr_tfidf_av_at[i].doc[0].lemma_

    lemmas_nccr_tfidf_av_ch = list(nlp.pipe(lf_input['tfidf_keywords_av_ch']))
    for i in range(len(lemmas_nccr_tfidf_av_ch)):
        lemmas_nccr_tfidf_av_ch[i] = lemmas_nccr_tfidf_av_ch[i].doc[0].lemma_

    lemmas_nccr_tfidf_av_de = list(nlp.pipe(lf_input['tfidf_keywords_av_de']))
    for i in range(len(lemmas_nccr_tfidf_av_de)):
        lemmas_nccr_tfidf_av_de[i] = lemmas_nccr_tfidf_av_de[i].doc[0].lemma_

    lemmas_nccr_tfidf_glob_at = list(nlp.pipe(lf_input['tfidf_keywords_global_at']))
    for i in range(len(lemmas_nccr_tfidf_glob_at)):
        lemmas_nccr_tfidf_glob_at[i] = lemmas_nccr_tfidf_glob_at[i].doc[0].lemma_

    lemmas_nccr_tfidf_glob_ch = list(nlp.pipe(lf_input['tfidf_keywords_global_ch']))
    for i in range(len(lemmas_nccr_tfidf_glob_ch)):
        lemmas_nccr_tfidf_glob_ch[i] = lemmas_nccr_tfidf_glob_ch[i].doc[0].lemma_

    lemmas_nccr_tfidf_glob_de = list(nlp.pipe(lf_input['tfidf_keywords_global_de']))
    for i in range(len(lemmas_nccr_tfidf_glob_de)):
        lemmas_nccr_tfidf_glob_de[i] = lemmas_nccr_tfidf_glob_de[i].doc[0].lemma_

    lemmas_nccr_chi2_glob = list(nlp.pipe(lf_input['chi2_keywords_global']))
    for i in range(len(lemmas_nccr_chi2_glob)):
        lemmas_nccr_chi2_glob[i] = lemmas_nccr_chi2_glob[i].doc[0].lemma_

    lemmas_nccr_chi2_glob_at = list(nlp.pipe(lf_input['chi2_keywords_at']))
    for i in range(len(lemmas_nccr_chi2_glob_at)):
        lemmas_nccr_chi2_glob_at[i] = lemmas_nccr_chi2_glob_at[i].doc[0].lemma_

    lemmas_nccr_chi2_glob_ch = list(nlp.pipe(lf_input['chi2_keywords_ch']))
    for i in range(len(lemmas_nccr_chi2_glob_ch)):
        lemmas_nccr_chi2_glob_ch[i] = lemmas_nccr_chi2_glob_ch[i].doc[0].lemma_

    lemmas_nccr_chi2_glob_de = list(nlp.pipe(lf_input['chi2_keywords_de']))
    for i in range(len(lemmas_nccr_chi2_glob_de)):
        lemmas_nccr_chi2_glob_de[i] = lemmas_nccr_chi2_glob_de[i].doc[0].lemma_

    # LF based on NCCR-constructed keyword lemmas
    @labeling_function(pre=[spacy_preprocessor])
    def lf_lemmas_nccr_tfidf_av(x):
        # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
        return POP if any(keyword in x.doc_lemmas for keyword in lemmas_nccr_tfidf) else ABSTAIN

    # LF based on NCCR global-constructed keywords
    @labeling_function(pre=[spacy_preprocessor])
    def lf_lemmas_nccr_tfidf_glob(x):
        # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
        return POP if any(keyword in x.doc_lemmas for keyword in lemmas_nccr_tfidf_glob) else ABSTAIN

    # LF based on NCCR country-constructed keywords (av)
    @labeling_function(pre=[spacy_preprocessor])
    def lf_lemmas_nccr_tfidf_av_country(x):
        if x.Sample_Country == 'au':
            # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
            return POP if any(keyword in x.doc_lemmas for keyword in lemmas_nccr_tfidf_av_at) else ABSTAIN

        elif x.Sample_Country == 'cd':
            # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
            return POP if any(keyword in x.doc_lemmas for keyword in lemmas_nccr_tfidf_av_ch) else ABSTAIN

        elif x.Sample_Country == 'de':
            # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
            return POP if any(keyword in x.doc_lemmas for keyword in lemmas_nccr_tfidf_av_de) else ABSTAIN

    # LF based on NCCR country-constructed keywords (global)
    @labeling_function(pre=[spacy_preprocessor])
    def lf_lemmas_nccr_tfidf_global_country(x):
        if x.Sample_Country == 'au':
            # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
            return POP if any(
                keyword in x.doc_lemmas for keyword in lemmas_nccr_tfidf_glob_at) else ABSTAIN

        elif x.Sample_Country == 'cd':
            # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
            return POP if any(
                keyword in x.doc_lemmas for keyword in lemmas_nccr_tfidf_glob_ch) else ABSTAIN

        elif x.Sample_Country == 'de':
            # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
            return POP if any(
                keyword in x.doc_lemmas for keyword in lemmas_nccr_tfidf_glob_de) else ABSTAIN

    # LF based on NCCR global-constructed keywords (chi2)
    @labeling_function(pre=[spacy_preprocessor])
    def lf_lemmas_nccr_chi2_glob(x):
        # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
        return POP if any(keyword in x.doc_lemmas for keyword in lemmas_nccr_chi2_glob) else ABSTAIN

    # LF based on NCCR country-constructed keywords (chi2)
    @labeling_function(pre=[spacy_preprocessor])
    def lf_lemmas_nccr_chi2_country(x):
        if x.Sample_Country == 'au':
            # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
            return POP if any(keyword in x.doc_lemmas for keyword in lemmas_nccr_chi2_glob_at) else ABSTAIN

        elif x.Sample_Country == 'cd':
            # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
            return POP if any(keyword in x.doc_lemmas for keyword in lemmas_nccr_chi2_glob_ch) else ABSTAIN

        elif x.Sample_Country == 'de':
            # Return a label of POP if keyword lemma in content, otherwise ABSTAIN
            return POP if any(keyword in x.doc_lemmas for keyword in lemmas_nccr_chi2_glob_de) else ABSTAIN

    # c) External Knowledge-based Labeling

    # LF based on party position estimated in CHES
    ches_14 = lf_input_ches['ches_14']
    ches_17 = lf_input_ches['ches_17']
    ches_19 = lf_input_ches['ches_19']

    @labeling_function()
    def lf_party_position_ches_pop(x):
        if x.year < 2014:
            return ABSTAIN

        elif 2014 <= x.year < 2017:
            if x.party in ches_14[x.Sample_Country]['pop']:
                return POP
            else:
                return ABSTAIN

        elif 2017 <= x.year < 2019:
            if x.party in ches_17[x.Sample_Country]['pop']:
                return POP
            else:
                return ABSTAIN

        elif 2019 <= x.year:
            if x.party in ches_19[x.Sample_Country]['pop']:
                return POP
            else:
                return ABSTAIN

    @labeling_function()
    def lf_party_position_ches_nonpop(x):
        if x.year < 2014:
            return ABSTAIN

        elif 2014 <= x.year < 2017:
            if x.party in ches_14[x.Sample_Country]['nonpop']:
                return NONPOP
            else:
                return ABSTAIN

        elif 2017 <= x.year < 2019:
            if x.party in ches_17[x.Sample_Country]['nonpop']:
                return NONPOP
            else:
                return ABSTAIN

        elif 2019 <= x.year:
            if x.party in ches_19[x.Sample_Country]['nonpop']:
                return NONPOP
            else:
                return ABSTAIN

    # d) DEP-based Labeling:

    # Initialize Dependency Matcher
    dep_matcher = DEP_Matcher(vocab=nlp_label.vocab).add_patterns()

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

    # Define list of lfs to use
    list_lfs = [lf_keywords_schwarzbozl,
                lf_keywords_rooduijn,

                lf_lemma_schwarzbozl,
                lf_lemma_rooduijn,

                lf_keywords_nccr_tfidf_av,
                lf_keywords_nccr_tfidf_glob,
                lf_keywords_nccr_tfidf_av_country,
                lf_keywords_nccr_tfidf_global_country,
                lf_keywords_nccr_chi2_glob,
                lf_keywords_nccr_chi2_country,

                lf_lemmas_nccr_tfidf_av,
                lf_lemmas_nccr_tfidf_glob,
                lf_lemmas_nccr_tfidf_av_country,
                lf_lemmas_nccr_tfidf_global_country,
                lf_lemmas_nccr_chi2_glob,
                lf_lemmas_nccr_chi2_country,

                lf_dep_pop_sv,
                lf_dep_pop_vo,
                lf_dep_pop_v,
                lf_dep_pop_s,
                lf_dep_pop_o,

                lf_dep_nonpop_sv,
                lf_dep_nonpop_vneg,
                lf_dep_nonpop_v,
                lf_dep_nonpop_s,

                lf_party_position_ches_pop,
                lf_party_position_ches_nonpop]

    return list_lfs

    # todo: transformation functions (e.g. https://www.snorkel.org/use-cases/02-spam-data-augmentation-tutorial)

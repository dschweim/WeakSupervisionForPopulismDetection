import spacy
import time
import itertools
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from collections import Counter

from util import extract_parsed_lemmas


class Dict_Generator:
    def __init__(
            self,
            data_path: str,
            output_path: str,
            spacy_model: str
    ):
        """
        Class to create dictionaries from labelled data
        :param data_path: path to data input
        :type data_path: str
        :param data_path: path to data output
        :type data_path: str
        :param spacy_model: used trained Spacy pipeline
        :type: str
        """

        self.data_path = data_path
        self.output_path = output_path
        self.nlp_full = spacy.load(spacy_model)

    @staticmethod
    def __custom_dict_tokenizer(text: str):
        """
        Tokenize text and remove stopwords + punctuation
        :param text: Text to tokenize
        :type text: str
        :return: Returns preprocessed Text
        :rtype:  str
        """

        german_stop_words = stopwords.words('german')  # Define stopwords
        text_tok = word_tokenize(text)  # Tokenize
        text_tok_sw = [word for word in text_tok if not word in german_stop_words]  # Remove stopwords
        text_tok_sw_alphanum = [word for word in text_tok_sw if word.isalnum()]  # Remove non-alphanumeric characters
        return text_tok_sw_alphanum

    def generate_tfidf_dict(self, df: pd.DataFrame, n_words: int):
        """
        Calculate tf-idf scores of docs and return top n words
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param n_words: Number of words to be included
        :type n_words: float
        :return: Returns df of dictionary words
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_dict_tokenizer, lowercase=False)

        # Fit vectorizer on whole corpus
        vectorizer.fit(df['wording_segments'])

        # CALCULATE TF-IDF SCORES OF POP CLASSIFIED DOCS
        df_pop = df.loc[df['POPULIST'] == 1]
        # Transform subcorpus labelled as POP
        tfidf_pop_vector = vectorizer.transform(df_pop['wording_segments']).toarray()

        # Map tf-idf scores to words in the vocab with separate column for each doc
        wordlist = pd.DataFrame({'term': vectorizer.get_feature_names()})

        # list = pd.DataFrame()
        for i in range(len(tfidf_pop_vector)):
            wordlist[i] = tfidf_pop_vector[i]

        # Set words as index
        wordlist.set_index('term')

        # Calculate average tf-idf over all docs
        wordlist['average_tfidf'] = wordlist.mean(axis=1)

        # Sort by average tf-idf
        wordlist.sort_values(by='average_tfidf', ascending=False, inplace=True)

        # Retrieve specified top n_words entries
        tfidf_dict = wordlist[:n_words][['term', 'average_tfidf']]

        # Save dict to disk
        tfidf_dict.to_csv(f'{self.output_path}\\tfidf_dict.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished tf-idf dict generation')

        return tfidf_dict

    def generate_tfidf_dict_per_country(self, df: pd.DataFrame, n_words: int):
        """
        Calculate tf-idf scores of docs per country and return top n words
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param n_words: Number of words to be included
        :type n_words: float
        :return: RReturns df of dictionary words
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_dict_tokenizer, lowercase=False)

        # Group data by country
        df_country_grpd = df.groupby('Sample_Country')

        # Initialize dict
        tfidf_dict_per_country = {}

        # Calculate tfidf dictionary per country
        for country, df_country in df_country_grpd:

            # Fit vectorizer on current corpus
            vectorizer.fit(df_country['wording_segments'])

            # CALCULATE TF-IDF SCORES OF POP CLASSIFIED DOCS
            df_country_pop = df_country.loc[df_country['POPULIST'] == 1]

            # Transform subcorpus labelled as POP
            tfidf_pop_vector = vectorizer.transform(df_country_pop['wording_segments']).toarray()

            # Map tf-idf scores to words in the vocab with separate column for each doc
            wordlist = pd.DataFrame({'term': vectorizer.get_feature_names()})

            # list = pd.DataFrame()
            for i in range(len(tfidf_pop_vector)):
                wordlist[i] = tfidf_pop_vector[i]

            # Set words as index
            wordlist.set_index('term')

            # Calculate average tf-idf over all docs
            wordlist['average_tfidf'] = wordlist.mean(axis=1)

            # Sort by average tf-idf
            wordlist.sort_values(by='average_tfidf', ascending=False, inplace=True)

            # Retrieve specified top n_words entries
            tfidf_dict = wordlist[:n_words][['term', 'average_tfidf']]

            # Append to country-specific dict to global dict
            tfidf_dict_per_country[country] = tfidf_dict

        # Save dict to disk
        tfidf_dict_per_country_au = tfidf_dict_per_country['au']
        tfidf_dict_per_country_ch = tfidf_dict_per_country['cd']
        tfidf_dict_per_country_de = tfidf_dict_per_country['de']
        tfidf_dict_per_country_au.to_csv(f'{self.output_path}\\tfidf_dict_per_country_au.csv', index=True)
        tfidf_dict_per_country_ch.to_csv(f'{self.output_path}\\tfidf_dict_per_country_ch.csv', index=True)
        tfidf_dict_per_country_de.to_csv(f'{self.output_path}\\tfidf_dict_per_country_de.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished tf-idf dict per country generation')

        return tfidf_dict_per_country

    def generate_global_tfidf_dict(self, df: pd.DataFrame, n_words: int):
        """
        Calculate tf-idf scores of docs and return top n words
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param n_words: Number of words to be included
        :type n_words: float
        :return: Returns df of dictionary words
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_dict_tokenizer, lowercase=False)

        # Generate two docs from corpus (POP and NON-POP)
        df_pop = df.loc[df['POPULIST'] == 1]
        df_nonpop = df.loc[df['POPULIST'] != 1]

        # Concatenate content of corpus
        content_pop = ' '.join(df_pop["wording_segments"])
        content_nonpop = ' '.join(df_nonpop["wording_segments"])

        # Generate global dataframe with two docs 'POP' and 'NONPOP'
        df_global = pd.DataFrame({'ID': ['df_pop', 'df_nonpop'],
                                  'Wording_combined': [content_pop, content_nonpop]})

        # Fit vectorizer on POP and NONPOP corpus
        vectorizer.fit(df_global['Wording_combined'])

        ## Calculate tf-idf scores of POP-corpus
        # Transform subcorpus labelled as POP
        tfidf_pop_vector = vectorizer.transform(df_global.loc[df_global['ID'] == 'df_pop'].Wording_combined).toarray()

        # Map tf-idf scores to words in the vocab with separate column for each doc
        wordlist = pd.DataFrame({'term': vectorizer.get_feature_names()})

        # list = pd.DataFrame()
        for i in range(len(tfidf_pop_vector)):
            wordlist[i] = tfidf_pop_vector[i]

        # Set words as index
        wordlist.set_index('term')

        # Sort by tf-idf
        wordlist.rename(columns={0: "tfidf"}, inplace=True)
        wordlist.sort_values(by='tfidf', ascending=False, inplace=True)

        # Retrieve specified top n_words entries
        tfidf_dict_global = wordlist[:n_words]

        # Save dict to disk
        tfidf_dict_global.to_csv(f'{self.output_path}\\tfidf_dict_global.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished tf-idf dict global generation')

        return tfidf_dict_global

    def generate_global_chisquare_dict(self, df: pd.DataFrame, confidence: float, n_words: int):
        """
        Calculate chi-square values of words and return top n words with value above critical value
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param confidence: level of statistical confidence
        :type confidence: float
        :param n_words: Number of words to be included
        :type n_words: float
        :return: Returns df of dictionary words
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = CountVectorizer(lowercase=False)

        # Generate two docs from corpus (POP and NON-POP)
        df_pop = df.loc[df['POPULIST'] == 1]
        df_nonpop = df.loc[df['POPULIST'] != 1]

        # Concatenate content of corpus
        content_pop = ' '.join(df_pop["wording_segments"])
        content_nonpop = ' '.join(df_nonpop["wording_segments"])

        # Generate global dataframe with two docs 'POP' and 'NONPOP'
        df_global = pd.DataFrame({'ID': ['df_pop', 'df_nonpop'],
                                  'Wording_combined': [content_pop, content_nonpop]})

        # Fit vectorizer on POP and NONPOP corpus
        vectorizer.fit(df_global['Wording_combined'])

        # Retrieve word counts of POP and NONPOP subsets
        count_vector = vectorizer.transform(df_global.Wording_combined).toarray()

        # Retrieve word counts of POP subset (S)
        count_pop_vector = count_vector[:][0]
        # Retrieve word counts of NONPOP subset (R)
        count_nonpop_vector = count_vector[:][1]

        # Retrieve total number of words for both corpora
        words_pop = count_pop_vector.sum()
        words_nonpop = count_nonpop_vector.sum()

        # Generate table with counts per word
        count_table = pd.DataFrame({'term': vectorizer.get_feature_names(),
                                    'popcount': count_pop_vector,
                                    'nonpopcount': count_nonpop_vector})

        # Only consider words with count of at least 5
        count_table = count_table.loc[(count_table['popcount'] >= 5) & (count_table['nonpopcount'] >= 5)]

        # Create empty dataframe for result
        result_table = pd.DataFrame()

        # for each word calculate chi-square statistics
        for index, word in count_table.iterrows():
            obs_freq_a = word.popcount
            obs_freq_b = word.nonpopcount
            obs_freq_c = words_pop - obs_freq_a
            obs_freq_d = words_nonpop - obs_freq_a

            # Define contingency table
            obs = np.array([[obs_freq_a, obs_freq_b],
                            [obs_freq_c, obs_freq_d]])

            # Calculate chi2, p, dof and ex
            chi2_word, p, dof, ex = chi2_contingency(obs)
            # Extract critical value dependent on confidence and dof
            critical = chi2.ppf(confidence, dof)

            # keep words where chi2 higher than critical value and correspond to pop corpus
            if chi2_word > critical:
                # Check whether word is dependent on pop corpus
                ratio_pop = obs_freq_a / obs_freq_b
                ratio_nonpop = (obs_freq_a + obs_freq_c) / (obs_freq_a + obs_freq_d)

                # If it is dependent on pop corpus, add to results
                if ratio_pop > ratio_nonpop:
                    chisquare_table = pd.DataFrame({'term': [word.term],
                                                    'chisquare': [chi2_word]})

                    # Append to result table
                    result_table = result_table.append(chisquare_table)

        # Sort by chi_square
        result_table.sort_values(by='chisquare', ascending=False, inplace=True)

        # Retrieve specified top n_words entries
        chisquare_dict = result_table[:n_words]

        # Save dict to disk
        chisquare_dict.to_csv(f'{self.output_path}\\chisquare_dict_global.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished chisquare dict global generation')

        return chisquare_dict

    def generate_chisquare_dict_per_country(self, df: pd.DataFrame, confidence: float, n_words: int):
        """
        Calculate chi-square values of words per country and return top n words with value above critical value
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param confidence: level of statistical confidence
        :type confidence: float
        :param n_words: Number of words to be included
        :type n_words: float
        :return: Returns df of dictionary words
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = CountVectorizer(lowercase=False)

        # Group data by country
        df_country_grpd = df.groupby('Sample_Country')

        # Initialize dict
        chisquare_dict_per_country = {}

        # Calculate tfidf dictionary per country
        for country, df_country in df_country_grpd:

            # Generate two docs from corpus (POP and NON-POP)
            df_pop = df_country.loc[df_country['POPULIST'] == 1]
            df_nonpop = df_country.loc[df_country['POPULIST'] != 1]

            # Concatenate content of corpus
            content_pop = ' '.join(df_pop["wording_segments"])
            content_nonpop = ' '.join(df_nonpop["wording_segments"])

            # Generate global dataframe with two docs 'POP' and 'NONPOP'
            df_global = pd.DataFrame({'ID': ['df_pop', 'df_nonpop'],
                                      'Wording_combined': [content_pop, content_nonpop]})

            # Fit vectorizer on POP and NONPOP corpus
            vectorizer.fit(df_global['Wording_combined'])

            # Retrieve word counts of POP and NONPOP subsets
            count_vector = vectorizer.transform(df_global.Wording_combined).toarray()

            # Retrieve word counts of POP subset (S)
            count_pop_vector = count_vector[:][0]
            # Retrieve word counts of NONPOP subset (R)
            count_nonpop_vector = count_vector[:][1]

            # Retrieve total number of words for both corpora
            words_pop = count_pop_vector.sum()
            words_nonpop = count_nonpop_vector.sum()

            # Generate table with counts per word
            count_table = pd.DataFrame({'term': vectorizer.get_feature_names(),
                                        'popcount': count_pop_vector,
                                        'nonpopcount': count_nonpop_vector})

            # Only consider words with count of at least 5
            count_table = count_table.loc[(count_table['popcount'] >= 5) & (count_table['nonpopcount'] >= 5)]

            # Create empty dataframe for result
            result_table_country = pd.DataFrame()

            # for each word calculate chi-square statistics
            for index, word in count_table.iterrows():
                obs_freq_a = word.popcount
                obs_freq_b = word.nonpopcount
                obs_freq_c = words_pop - obs_freq_a
                obs_freq_d = words_nonpop - obs_freq_a

                # Define contingency table
                obs = np.array([[obs_freq_a, obs_freq_b],
                                [obs_freq_c, obs_freq_d]])

                # Calculate chi2, p, dof and ex
                chi2_word, p, dof, ex = chi2_contingency(obs)
                # Extract critical value dependent on confidence and dof
                critical = chi2.ppf(confidence, dof)

                # keep words where chi2 higher than critical value and correspond to pop corpus
                if chi2_word > critical:
                    # Check whether word is dependent on pop corpus
                    ratio_pop = obs_freq_a / obs_freq_b
                    ratio_nonpop = (obs_freq_a + obs_freq_c) / (obs_freq_a + obs_freq_d)

                    # If it is dependent on pop corpus, add to results
                    if ratio_pop > ratio_nonpop:
                        chisquare_table = pd.DataFrame({'term': [word.term],
                                                        'chisquare': [chi2_word]})

                        # Append to result table
                        result_table_country = result_table_country.append(chisquare_table)

            # Sort by chi-square
            result_table_country.sort_values(by='chisquare', ascending=False, inplace=True)

            # Retrieve specified top n_words entries
            chisquare_dict = result_table_country[:n_words]

            # Append to country-specific dict to global dict
            chisquare_dict_per_country[country] = chisquare_dict

        # Save dict to disk
        chisquare_dict_per_country_au = chisquare_dict_per_country['au']
        chisquare_dict_per_country_ch = chisquare_dict_per_country['cd']
        chisquare_dict_per_country_de = chisquare_dict_per_country['de']
        chisquare_dict_per_country_au.to_csv(f'{self.output_path}\\chisquare_dict_per_country_au.csv',
                                             index=True)
        chisquare_dict_per_country_ch.to_csv(f'{self.output_path}\\chisquare_dict_per_country_ch.csv',
                                             index=True)
        chisquare_dict_per_country_de.to_csv(f'{self.output_path}\\chisquare_dict_per_country_de.csv',
                                             index=True)

        end = time.time()
        print(end - start)
        print('finished chisquare dict per country generation')

        return chisquare_dict_per_country

    def generate_chisquare_dict_antielite(self, df: pd.DataFrame, preprocessed: bool):
        """
        Calculate tf-idf scores of docs and return top words
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :return: Returns df of dictionary words
        :rtype:  DataFrame
        """

        start = time.time()

        # Generate spacy docs from corpus if necessary
        if not preprocessed:
            df['wording_segments_doc'] = list(self.nlp_full.pipe(df['wording_segments']))

        # Generate two docs from corpus (POP and NONPOP) #todo: check whether split correct
        df_pop = df.loc[(df['POPULIST'] == 1)]
        df_rest = df[~df.index.isin(df_pop.index)]

        # Generate dict for both corpora separately
        #full_dict_ae = df_pop['wording_segments_doc'].apply(lambda x: self.__extract_dep_triples(x, True))
        #lemma_ae = df_pop['wording_segments_doc'].apply(lambda x:  extract_parsed_lemmas(x))
        #flat_dict_ae = df_pop['wording_segments_doc'].apply(lambda x: self.__extract_dep_triples(x, False))

        flat_dict_ae = df_pop['wording_segments_doc'].apply(lambda x: self.__extract_dep_tuples(x))


        #dict_rest = df_rest['wording_segments_doc'].apply(lambda x: self.__extract_dep_triples(x, True))


        # Generate triples (subj, obj, neg + verb, pred) of full corpus AE and NOT AE
        def generate_corpus_triples(series):

            corpus_triples = []

            for index, value in series.items():

                # Iterate over dicts (i.e. number of sentences)
                for elem in value:
                    # Skip None values
                    if isinstance(elem, dict):
                        current_val = list(elem.values())

                        subjs = ', '.join(current_val[0])
                        preds = ', '.join(current_val[1])
                        negs = ', '.join(current_val[2])
                        verbs = ', '.join(current_val[3])

                        # objs = ', '.join(current_val[4])

                        current_triple = (subjs, preds, negs, verbs)

                        corpus_triples.append(current_triple)

            # Generate df
            triples_df = pd.DataFrame({'triple': Counter(corpus_triples).keys(),  # get unique values of triples
                                       'count': Counter(corpus_triples).values()})  # get the elements' frequency

            return triples_df #return without duplicates +  get count

        ae_triples_df = generate_corpus_triples(flat_dict_ae)

        # Concatenate content of corpus
        content_ae = ' '.join(df_pop["wording_segments"])
        content_rest = ' '.join(df_rest["wording_segments"])

        # Append empty counts
        ae_triples_df["count_ae"] = np.nan
        ae_triples_df["count_rest"] = np.nan

        # # Count number of times, each triple occurs in corpus ae/rest respectively
        # for index, row in ae_triples_df.iterrows:
        #     content_ae = str.count(row.triple)



        # Append triples with high enough chisquare to dict
        chisquare_ae_dict = {}

        # Save dict to disk
        #tfidf_ae_dict.to_csv(f'{self.output_path}\\tfidf_antielite_dict.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished tf-idf antielite dict generation')

        return chisquare_ae_dict

    # todo:Extract tuples from sentence (subjects, pred)
    def __extract_dep_tuples(self, segment):

        tuples_dict_list = []

        # Get HEAD tokens
        heads = [token for token in segment if token.head == token]

        # Iterate over heads
        for head in heads:

            lemmas = []
            subj_list = []
            pred_list = []
            obj_list = []
            neg_list = []

            current_sent = head.sent

            # Case 1: head is verb
            for token in current_sent:
                lemmas.append((token.lemma_.lower(), token.pos_, token.dep_, token.head.text))

            if head.pos_ == 'VERB':

                # Add head to verb list
                pred_list.append(head.lemma_.lower())

                for child in head.children:

                    # Extract additional Verb components and auxiliaries and predicates
                    # if child.pos_ in ['VERB']:
                    #     pred_list.append(child.lemma_.lower())

                    if child.dep_ == 'pd':
                        pred_list.append(child.lemma_.lower())

                    # Check if sentence contains negation
                    elif child.dep_ == 'neg':
                        neg_list.append(child.lemma_.lower())

                    # Extract subjects
                    elif (child.dep_ == 'sb') & (child.pos_ not in ['AUX', 'VERB']):
                        subj_list.append(child.lemma_.lower())

                    # Extract objects
                    elif (child.dep_ in ['oa', 'oc', 'da', 'og', 'op', 'ag']) & (child.pos_ not in ['AUX', 'VERB']):
                        obj_list.append(child.lemma_.lower())

            ## OTHER CASES OF ROOT
            elif head.pos_ == 'AUX':

                for child in head.children:

                    # Extract additional Verb components and auxiliaries and predicates
                    if child.pos_ in ['VERB']:
                        pred_list.append(child.lemma_.lower())

                    if child.dep_ == 'pd':
                        pred_list.append(child.lemma_.lower())

                    # Check if sentence contains negation
                    if child.dep_ == 'neg':
                        neg_list.append(child.lemma_.lower())

                    # Extract subjects
                    if (child.dep_ == 'sb') & (child.pos_ not in ['AUX', 'VERB']):
                        subj_list.append(child.lemma_.lower())

                    # Extract objects
                    if (child.dep_ in ['oa', 'da', 'og', 'ag']) & (child.pos_ not in ['AUX', 'VERB']):
                        obj_list.append(child.lemma_.lower())

            elif head.pos_ == 'NOUN':

                # Add head to subject list
                subj_list.append(head.lemma_.lower())

                for child in head.children:
                    # Extract additional Verb components and auxiliaries and predicates
                    if child.pos_ in ['VERB']:
                        pred_list.append(child.lemma_.lower())

                    if child.dep_ == 'pd':
                        pred_list.append(child.lemma_.lower())

                    # Check if sentence contains negation
                    if child.dep_ == 'neg':
                        neg_list.append(child.lemma_.lower())

                    # Extract subjects
                    if (child.dep_ == 'sb') & (child.pos_ not in ['AUX', 'VERB']):
                        subj_list.append(child.lemma_.lower())

                    # Extract objects
                    if (child.dep_ in ['oa', 'da', 'og', 'ag']) & (child.pos_ not in ['AUX', 'VERB']):
                        obj_list.append(child.lemma_.lower())

            # if lists are empty, return None
            if (not subj_list) & (not pred_list) & (not obj_list) & (not neg_list):
                tuples_dict = None

            else:
                tuples_dict = {'subjects': subj_list,
                                'predicates': pred_list,
                               'negations': neg_list,
                               'objects': obj_list}

            # Generate list with dict for each head
            tuples_dict_list.append(tuples_dict)

        # if lists are empty, return None
        if (not subj_list) & (not pred_list) & (not obj_list) & (not neg_list):
            return []
        # else
        else:
            return tuples_dict_list




    def __extract_dep_triples(self, segment, fulldetail: bool):
        """
        Function to retrieve triples from segment
        :param segment:
        :return:
        """

        SUBJECTS = ['sb']
        OBJECTS = ['oa', 'oc', 'og', 'op', 'da', 'ag']
        PREDICATES = ['pd']
        VERBCOMPONENTS = ['svp']  # separable verb prefix
        NEGATIONS = ['ng']

        triples_dict_list = []

        # Get HEAD tokens
        heads = [token for token in segment if token.head == token]

        # Iterate over heads
        for head in heads:

            obj_list = []
            subj_list = []
            pred_list = []
            verb_list = []
            neg_list = []

            current_sent = head.sent

            # Case 1: head is verb-type (verb, aux)
            if (head.pos_ == 'VERB') or (head.pos_ == 'AUX'):

                # Add head to verb list
                if fulldetail:
                    verb_list.append((head.lemma_.lower(), head.pos_, head.dep_))
                else:
                    verb_list.append(head.lemma_.lower())

                for child in head.children:
                    # Extract additional Verb components and auxiliaries
                    if child.dep_ in VERBCOMPONENTS:
                        if fulldetail:
                            verb_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            verb_list.append(child.lemma_.lower())

                    if child.pos_ in ['AUX', 'VERB']:
                        if fulldetail:
                            verb_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            verb_list.append(child.lemma_.lower())

                    # Check if sentence contains negation
                    if child.dep_ in NEGATIONS:
                        if fulldetail:
                            neg_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            neg_list.append(child.lemma_.lower())

                    # Extract objects
                    if (child.dep_ in OBJECTS) & (child.pos_ not in ['AUX', 'VERB']):
                        if fulldetail:
                            obj_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            obj_list.append(child.lemma_.lower())

                    # Extract subjects
                    if (child.dep_ in SUBJECTS) & (child.pos_ not in ['AUX', 'VERB']):
                        if fulldetail:
                            subj_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            subj_list.append(child.lemma_.lower())

                    # Extract predicates
                    if child.dep_ in PREDICATES:
                        if fulldetail:
                            pred_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            pred_list.append(child.lemma_.lower())

            # Case 2: head is noun
            if head.pos_ == 'NOUN':
                # Add head to subject list #todo: is subject or object???
                if fulldetail:
                    subj_list.append((head.lemma_.lower(), head.pos_, head.dep_))
                else:
                    subj_list.append(head.lemma_.lower())

                for child in head.children:

                    # Extract additional Verb components and auxiliaries
                    if child.dep_ in VERBCOMPONENTS:
                        if fulldetail:
                            verb_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            verb_list.append(child.lemma_.lower())
                    if child.pos_ == 'AUX':
                        if fulldetail:
                            verb_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            verb_list.append(child.lemma_.lower())
                    # Check if sentence contains negation
                    if child.dep_ in NEGATIONS:
                        if fulldetail:
                            neg_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            neg_list.append(child.lemma_.lower())

                    # Extract objects
                    if (child.dep_ in OBJECTS) & (child.pos_ not in ['AUX', 'VERB']):
                        if fulldetail:
                            obj_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            obj_list.append(child.lemma_.lower())

                    # Extract subjects
                    if (child.dep_ in SUBJECTS) & (child.pos_ not in ['AUX', 'VERB']):
                        if fulldetail:
                            subj_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            subj_list.append(child.lemma_.lower())

                    # Extract predicates
                    if child.dep_ in PREDICATES:
                        if fulldetail:
                            pred_list.append((child.lemma_.lower(), child.pos_, child.dep_))
                        else:
                            pred_list.append(child.lemma_.lower())

            # if lists are empty, return None
            if (not obj_list) & (not subj_list) & (not verb_list) & (not neg_list) & (not pred_list):
                triples_dict = None

            else:
                triples_dict = {'subjects': subj_list,
                                'negations': neg_list,
                                'verbs': verb_list,
                                'predicates': pred_list,
                                'objects': obj_list}

            # Generate list with dict for each head
            triples_dict_list.append(triples_dict)

        # if lists are empty, return None
        if (not obj_list) & (not subj_list) & (not verb_list) & (not neg_list) & (not pred_list):
            return []
        # else
        else:
            return triples_dict_list



import spacy
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from util import extract_parsed_lemmas, extract_dep_tuples, get_all_svo_tuples
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
        self.spacy_model = spacy_model
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

    def generate_tfidf_av_dict(self, df: pd.DataFrame, n_words: int):
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
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_dict_tokenizer, lowercase=True)

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
        tfidf_dict.to_csv(f'{self.output_path}\\tfidf_dict_av.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished tf-idf dict av generation')

        return tfidf_dict

    def generate_tfidf_av_dict_per_country(self, df: pd.DataFrame, n_words: int):
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
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_dict_tokenizer, lowercase=True)

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
        tfidf_dict_per_country_au.to_csv(f'{self.output_path}\\tfidf_dict_av_per_country_au.csv', index=True)
        tfidf_dict_per_country_ch.to_csv(f'{self.output_path}\\tfidf_dict_av_per_country_ch.csv', index=True)
        tfidf_dict_per_country_de.to_csv(f'{self.output_path}\\tfidf_dict_av_per_country_de.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished tf-idf dict av per country generation')

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
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_dict_tokenizer, lowercase=True)

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

    # todo:
    # def generate_global_tfidf_dict_per_country(self, df: pd.DataFrame, n_words: int):
    #
    #     return None

    def generate_global_chisquare_dict(self, df: pd.DataFrame, confidence: float):
        """
        Calculate chi-square values of words and return top n words with value above critical value
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param confidence: level of statistical confidence
        :type confidence: float
        :return: Returns df of dictionary words
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = CountVectorizer(tokenizer=self.__custom_dict_tokenizer, lowercase=True)

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

        # Retrieve total number of words for both corpora #todo: maybe only for >5?
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
        chisquare_dict = result_table

        # Save dict to disk
        chisquare_dict.to_csv(f'{self.output_path}\\chisquare_dict_global.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished chisquare dict global generation')

        return chisquare_dict

    def generate_chisquare_dict_per_country(self, df: pd.DataFrame, confidence: float):
        """
        Calculate chi-square values of words per country and return top n words with value above critical value
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param confidence: level of statistical confidence
        :type confidence: float
        :return: Returns df of dictionary words
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = CountVectorizer(tokenizer=self.__custom_dict_tokenizer, lowercase=True)

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
            chisquare_dict = result_table_country

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
        print('finished chisquare dict global per country generation')

        return chisquare_dict_per_country

    def generate_chisquare_dep_dicts(self, df: pd.DataFrame, preprocessed: bool, confidence: float):
        """
        Calculate
        :param df: corpus from which to construct dict
        :type df:  DataFrame
        :param preprocessed: Indicator whether corpus has been preprocessed in current run with Spacy pipe
        :type preprocessed:  bool
        :param confidence: level of statistical confidence
        :type confidence: float
        :return: Returns df of dictionary words
        :rtype:  DataFrame
        """

        start = time.time()

        # Generate spacy docs from corpus
        df['wording_segments_doc'] = list(self.nlp_full.pipe(df['wording_segments']))

        # Generate two docs from corpus (POP and NONPOP)
        df_pop = df.loc[(df['POPULIST'] == 1)]
        df_nonpop = df[~df.index.isin(df_pop.index)]

        # Extract svo-triples per Segment for both corpora separately
        svo_tuples_pop = df_pop['wording_segments_doc'].apply(lambda x: extract_dep_tuples(x))
        svo_tuples_nonpop = df_nonpop['wording_segments_doc'].apply(lambda x: extract_dep_tuples(x))

        #Iterate over tuple combinations
        combinations = [{'subj': True, 'verb': True, 'verbprefix': True, 'obj': True, 'neg': True}, # svo + prefix + neg
                        {'subj': True, 'verb': True, 'verbprefix': True, 'obj': True, 'neg': False},  # svo + prefix
                        {'subj': True, 'verb': True, 'verbprefix': False, 'obj': True, 'neg': True},  # svo + neg
                        {'subj': True, 'verb': True, 'verbprefix': False, 'obj': True, 'neg': False},  # svo
                        {'subj': True, 'verb': True, 'verbprefix': False, 'obj': False, 'neg': True},  # sv + neg
                        {'subj': True, 'verb': True, 'verbprefix': False, 'obj': False, 'neg': False},  # sv
                        {'subj': False, 'verb': True, 'verbprefix': False, 'obj': True, 'neg': True},  # vo + neg
                        {'subj': False, 'verb': True, 'verbprefix': False, 'obj': True, 'neg': False},  # vo
                        {'subj': True, 'verb': False, 'verbprefix': False, 'obj': True, 'neg': False},  # so
                        {'subj': False, 'verb': True, 'verbprefix': False, 'obj': False, 'neg': True},  # v + neg
                        {'subj': False, 'verb': True, 'verbprefix': False, 'obj': False, 'neg': False},  # v
                        {'subj': True, 'verb': False, 'verbprefix': False, 'obj': False, 'neg': False},  # s
                        {'subj': False, 'verb': False, 'verbprefix': False, 'obj': True, 'neg': False}  # o
                        ]

        global_pop_dict = dict()
        global_nonpop_dict = dict()

        for combination in combinations:
            get_components = combination

            # Generate list of all distinct svo-triples and sort by their number of occurrences
            svo_tuples_pop_list = get_all_svo_tuples(svo_tuples_pop, get_components).sort_values(by='count', ascending=False)
            svo_tuples_nonpop_list = get_all_svo_tuples(svo_tuples_nonpop, get_components).sort_values(by='count', ascending=False)

            # Rename columns
            svo_tuples_pop_list.rename({'count': 'count_pop'}, axis=1, inplace=True)
            svo_tuples_nonpop_list.rename({'count': 'count_nonpop'}, axis=1, inplace=True)

            # Append both dfs
            svo_tuples_list = svo_tuples_pop_list.append(svo_tuples_nonpop_list)

            # Group df by tuple and aggregate counts
            svo_tuples_grpd = svo_tuples_list.groupby(by="tuple", dropna=False).agg({'count_pop': 'sum', 'count_nonpop': 'sum'})

            # Get total counts (POP + NONPOP) per tuple and reset index
            svo_tuples_grpd['count_total'] = svo_tuples_grpd.count_pop + svo_tuples_grpd.count_nonpop
            svo_tuples_grpd.reset_index(inplace=True)

            # Retrieve total number of tuples for both corpora
            tuples_pop = svo_tuples_grpd.count_pop.sum()
            tuples_nonpop = svo_tuples_grpd.count_nonpop.sum()

            # Only consider words with count of at least 5 #todo: remove at word count as well?
            svo_tuples_grpd = svo_tuples_grpd.loc[(svo_tuples_grpd['count_pop'] >= 5) & (svo_tuples_grpd['count_nonpop'] >= 5)]

            # Create empty dataframes for result
            chisquare_tuples_pop = pd.DataFrame()
            chisquare_tuples_nonpop = pd.DataFrame()

            # for each word calculate chi-square statistics
            for index, tuple in svo_tuples_grpd.iterrows():
                obs_freq_a = tuple.count_pop
                obs_freq_b = tuple.count_nonpop
                obs_freq_c = tuples_pop - obs_freq_a
                obs_freq_d = tuples_nonpop - obs_freq_a

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

                    # If it is dependent on pop corpus, add to pop dict , otherwhise to nonpop dict
                    chisquare_table = pd.DataFrame({'tuple': [tuple.tuple],
                                                    'chisquare': [chi2_word]})
                    if ratio_pop > ratio_nonpop:
                        # Append to result pop dict
                        chisquare_tuples_pop = chisquare_tuples_pop.append(chisquare_table)

                    else:
                        # Append to result nonpop dict
                        chisquare_tuples_nonpop = chisquare_tuples_nonpop.append(chisquare_table)

            # Sort by chi_square
            if len(chisquare_tuples_pop) > 0:
                chisquare_tuples_pop.sort_values(by='chisquare', ascending=False, inplace=True)
                global_pop_dict[str(combination)] = chisquare_tuples_pop.tuple.values
            else:
                global_pop_dict[str(combination)] = None

            if len(chisquare_tuples_nonpop) > 0:
                chisquare_tuples_nonpop.sort_values(by='chisquare', ascending=False, inplace=True)
                global_nonpop_dict[str(combination)] = chisquare_tuples_nonpop.tuple.values
            else:
                global_nonpop_dict[str(combination)] = None

        # only for debugging
        lemma_pop = df_pop['wording_segments_doc'].apply(lambda x: extract_parsed_lemmas(x))
        lemma_nonpop = df_nonpop['wording_segments_doc'].apply(lambda x: extract_parsed_lemmas(x))

        # Save dict to disk
        for combination in combinations:
            if global_pop_dict[str(combination)] is not None:
                global_pop_dict_com = global_pop_dict[str(combination)]
                pd.DataFrame(global_pop_dict_com).to_csv(
                    f'{self.output_path}\\chisquare_dep_pop_{str(combination.values())}_{confidence}.csv',
                    index=True)

        for combination in combinations:
            if global_nonpop_dict[str(combination)] is not None:
                global_nonpop_dict_com = global_nonpop_dict[str(combination)]
                pd.DataFrame(global_nonpop_dict_com).to_csv(
                    f'{self.output_path}\\chisquare_dep_nonpop_{str(combination.values())}_{confidence}.csv',
                    index=True)

        end = time.time()
        print(end - start)
        print('finished dependency-based dicts generation for pop- & nonpop-dimensions')

        return global_pop_dict, global_nonpop_dict



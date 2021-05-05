import os
import glob
import re
import time
import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class PCCR_Dataset:
    def __init__(
            self,
            data_path: str,
            output_path: str,
    ):
        """
        Class to create the NCCR data
        :param data_path:
        :type data_path:
        """

        self.data_path = data_path
        self.output_path = output_path

    def generate_labelled_nccr_corpus(self):
        """
        Generate labelled NCCR corpus by merging txt-fulltexts
        with their corresponding labels on text-level and join with full_issue and
        full_target
        :return: Labelled NCCR Corpus with all rows
        :rtype: DataFrame
        :return: Labelled NCCR Corpus only with rows that contain Wording info
        :rtype: DataFrame
        """

        start = time.time()

        # Get filenames of texts
        os.chdir(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Texts\\Texts')
        files = [i for i in glob.glob("*.txt")]

        # Create empty dataframe for texts
        df = pd.DataFrame()

        # Read every txt file in dir and add to df
        for file in files:
            tmp = open(file, encoding="ISO-8859-1").read()
            txt = pd.DataFrame({'ID': [file],
                                'text': [tmp]})
            df = df.append(txt)

        # Save concatenated texts
        df.to_csv(f'{self.output_path}\\NCCR_concatenated_texts.csv', index=True)

        # Import corpus with populism labels
        table_text = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Text_Table.txt', delimiter="\t",
                                 encoding="ISO-8859-1")

        # Join both dataframes
        df_combined = df.set_index('ID').join(table_text.set_index('ID'))

        # Filter on German files
        df_combined_de = df_combined[df_combined.Sample_Lang == 'Deutsch']

        # Find duplicates
        df_combined_de.reset_index(inplace=True)
        duplicates_list = df_combined_de[df_combined_de.duplicated(subset=['ID'], keep=False)]['ID']

        # Remove duplicates that do not belong to sample
        df_drop = df_combined_de.loc[((df_combined_de['Bemerkungen'] == 'Does not belong to the sample') |
                                      (df_combined_de['Bemerkungen'] == 'Does not belong to the sample / '))
                                     & df_combined_de['ID'].isin(duplicates_list)]

        df_combined_de = df_combined_de.drop(df_combined_de.index[[df_drop.index.values]])

        # todo: Remove remaining duplicates
        duplicates = df_combined_de[df_combined_de.duplicated(subset=['ID'], keep=False)]

        # Save created German corpus
        #df_combined_de.to_csv(f'{self.output_path}\\NCCR_labelled_corpus_DE.csv', index=True)

        # Merge combined df with full_target, full_issue
        #full_speaker = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Fulltext_Speaker.csv')
        full_issue = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Fulltext_Issue.csv')
        full_target = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Fulltext_Target.csv')
        table_text_issue = df_combined_de.set_index('ID').join(full_issue.set_index('ID'))
        table_text_issue['Source'] = 'Issue'
        table_text_target = df_combined_de.set_index('ID').join(full_target.set_index('ID'))
        table_text_target['Source'] = 'Target'
        table_text_combined = table_text_issue.append(table_text_target)
        table_text_combined.reset_index(inplace=True)

        # Remove rows with "UK" ID
        table_text_combined_de = table_text_combined[~table_text_combined['ID'].astype(str).str.startswith('uk')]

        # Filter on relevant columns
        table_text_combined_de = table_text_combined_de[['ID',
                                                          'POPULIST',
                                                          'POPULIST_PeopleCent', 'POPULIST_AntiElite', 'POPULIST_Sovereign',
                                                          'POPULIST_Advocative', 'POPULIST_Conflictive',
                                                          'ANTIPOPULIST',
                                                          'APOPULIST_PeopleCent', 'APOPULIST_AntiElite', 'APOPULIST_Sovereign',
                                                          'APOPULIST_Advocative', 'APOPULIST_Conflictive',
                                                          'Date', 'Sample_Type', 'Sample_Country',
                                                          'Wording', 'Fulltext',
                                                          'text', 'Source']]

        # Sort by ID
        table_text_combined_de.sort_values(by='ID', inplace=True)

        # Save created merged corpus
        table_text_combined_de.to_csv(f'{self.output_path}\\NCCR_combined_corpus_DE_wording_all.csv', index=True)

        # Exclude examples without wording
        table_text_combined_de_av = table_text_combined_de.dropna(subset=['Wording'])

        # Save created merged corpus
        table_text_combined_de_av.to_csv(f'{self.output_path}\\NCCR_combined_corpus_DE_wording_available.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished NCCR labelled corpus generation')

        return table_text_combined_de, table_text_combined_de_av

    def preprocess_corpus(self, df: pd.DataFrame, is_train: bool):
        """
        Preprocess text of corpus
        :param df: Dataset to preprocess
        :type df:  DataFrame
        :param is_train: Indicates if df is train set
        :type is_train:  boolean
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        start = time.time()

        if is_train:
            label = 'TRAIN'
        else:
            label = 'TEST'

        def preprocess_text(text):
            # Remove standard text info at beginning of text
            text = re.sub(r'^(\n|.)*--', '', text)

            # Remove linebreaks and extra spaces
            text = " ".join(text.split())

            # Remove some special characters (*) todo: instead remove every non-word/space/punctuation
            text = text.replace('*', '')

            return text

        df['text_prep'] = df['text'].apply(lambda x: preprocess_text(x))

        # Save pre-processed corpus
        df.to_csv(f'{self.output_path}\\NCCR_combined_corpus_DE_wording_available_{label}.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished dataset preprocessing for ' + label)

        return df

    def __custom_tokenizer(self, text):
        """
        Tokenize text and remove stopwords + punctuation
        :param text: Text to tokenize
        :type text: str
        :return: Returns preprocessed Text
        :rtype:  str
        """

        german_stop_words = stopwords.words('german')
        german_stop_words.append('fuer')

        text_tok = word_tokenize(text)  # Tokenize
        text_tok_sw = [word for word in text_tok if not word in german_stop_words]  # Remove stopwords
        text_tok_sw_alphanum = [word for word in text_tok_sw if word.isalnum()]  # Remove punctuation
        return text_tok_sw_alphanum

    def generate_tfidf_dict(self, df: pd.DataFrame, tfidf_threshold: float):
        """
        Calculate tf-idf scores of docs and return top n words with tfidf above threshold
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param tfidf_threshold: Min value for words to be considered in dict
        :type tfidf_threshold: float
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_tokenizer)

        # Fit vectorizer on whole corpus
        vectorizer.fit(df['text_prep'])

        # CALCULATE TF-IDF SCORES OF POP CLASSIFIED DOCS
        df_pop = df.loc[df['POPULIST'] == 1]
        # Transform subcorpus labelled as POP
        tfidf_pop_vector = vectorizer.transform(df_pop['text_prep']).toarray()

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
        tfidf_dict = wordlist.loc[wordlist['average_tfidf'] >= tfidf_threshold][['term', 'average_tfidf']]

        # Save dict to disk
        tfidf_dict.to_csv(f'{self.output_path}\\tfidf_dict.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished tf-idf dict generation')

        return tfidf_dict

    def generate_tfidf_dict_per_country(self, df: pd.DataFrame, tfidf_threshold: float):
        """
        Calculate tf-idf scores of docs per country and return top n words with tfidf above threshold
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param tfidf_threshold: Min value for words to be considered in dict
        :type tfidf_threshold: float
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_tokenizer)

        # Group data by country
        df_country_grpd = df.groupby('Sample_Country')

        # Initialize dict
        tfidf_dict_per_country = {}

        # Calculate tfidf dictionary per country
        for country, df_country in df_country_grpd:

            # Fit vectorizer on current corpus
            vectorizer.fit(df_country['text_prep'])

            # CALCULATE TF-IDF SCORES OF POP CLASSIFIED DOCS
            df_country_pop = df_country.loc[df_country['POPULIST'] == 1]

            # Transform subcorpus labelled as POP
            tfidf_pop_vector = vectorizer.transform(df_country_pop['text_prep']).toarray()

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
            tfidf_dict = wordlist.loc[wordlist['average_tfidf'] >= tfidf_threshold][['term', 'average_tfidf']]

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

    def generate_global_tfidf_dict(self, df: pd.DataFrame, tfidf_threshold: float):
        """
        Calculate tf-idf scores of docs and return top n words with tfidf above threshold
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param tfidf_threshold: Min value for words to be considered in dict
        :type tfidf_threshold: float
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_tokenizer)

        # Generate two docs from corpus (POP and NON-POP)
        df_pop = df.loc[df['POPULIST'] == 1]
        df_nonpop = df.loc[df['POPULIST'] != 1]

        # Concatenate content of corpus
        content_pop = ' '.join(df_pop["text_prep"])
        content_nonpop = ' '.join(df_nonpop["text_prep"])

        # Generate global dataframe with two docs 'POP' and 'NONPOP'
        df_global = pd.DataFrame({'ID': ['df_pop', 'df_nonpop'],
                                'text_prep': [content_pop, content_nonpop]})

        # Fit vectorizer on POP and NONPOP corpus
        vectorizer.fit(df_global['text_prep'])

        ## Calculate tf-idf scores of POP-corpus
        # Transform subcorpus labelled as POP
        tfidf_pop_vector = vectorizer.transform(df_global.loc[df_global['ID'] == 'df_pop'].text_prep).toarray()

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
        tfidf_dict_global = wordlist.loc[wordlist['tfidf'] >= tfidf_threshold]

        # Save dict to disk
        tfidf_dict_global.to_csv(f'{self.output_path}\\ tfidf_dict_global.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished tf-idf dict global generation')

        return tfidf_dict_global

import os
import glob
import spacy
import re
import time
import spacy
import nltk

import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class PCCR_Dataset:
    def __init__(
            self,
            data_path: str,
            output_path: str
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
        with their corresponding labels on text-level and save as csv
        :return:
        :rtype:
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

        df.to_csv(f'{self.output_path}\\NCCR_concatenated_texts.csv', index=True)

        # Import corpus with populism labels
        table_text = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Text_Table.txt', delimiter="\t",
                                 encoding="ISO-8859-1")

        # Filter on relevant columns
        table_text = table_text[['ID', 'POPULIST', 'POPULIST_PeopleCent', 'POPULIST_AntiElite', 'POPULIST_Sovereign',
                                 'POPULIST_Advocative', 'POPULIST_Conflictive', 'ANTIPOPULIST', 'APOPULIST_PeopleCent',
                                 'APOPULIST_AntiElite', 'APOPULIST_Sovereign', 'APOPULIST_Advocative',
                                 'APOPULIST_Conflictive', 'POP_Total', 'POP_Unchall', 'POP_Emph',
                                 'Main_Issue',
                                 'Author', 'Date',
                                 'Bemerkungen',
                                 'Sample_Lang',
                                 'Sample_Type']]

        # Join both dataframes
        df_combined = df.set_index('ID').join(table_text.set_index('ID'))

        # Save created corpus
        df_combined.to_csv(f'{self.output_path}\\labelled_nccr_corpus.csv', index=True)

        # Filter on German files
        df_combined_de = df_combined[df_combined.Sample_Lang == 'Deutsch']
        # Save created German corpus
        df_combined_de.to_csv(f'{self.output_path}\\labelled_nccr_corpus_DE.csv', index=True)

        # todo: check whether any ID occurs more than once

        end = time.time()
        print(end - start)
        print('finished NCCR labelled corpus generation')

        return df_combined_de

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

        nlp = spacy.load("de_core_news_sm")

        def preprocess_text(text):
            # Remove standard text info at beginning of text
            text = re.sub(r'((\n|.)*--)', '', text)

            # Remove linebreaks and extra spaces
            text = " ".join(text.split())

            # Remove some special characters (*) todo: instead remove every non-word/space/punctuation
            text = text.replace('*', '')

            # Split text into sentences
            # doc = nlp(text)
            # text = [sentence.text for sentence in doc.sents]

            return text

        df['text_prep'] = df['text'].apply(lambda x: preprocess_text(x))

        # Save pre-processed corpus
        df.to_csv(f'{self.output_path}\\labelled_nccr_corpus_DE_{label}.csv', index=True)

        end = time.time()
        print(end - start)
        print ('finished dataset preprocessing for ' + label)

        return df

    def generate_tfidf_dict(self, df: pd.DataFrame, tfidf_threshold: int):
        """
        Calculate tf-idf scores of docs and return top n words that
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param tfidf_threshold: Min value for words to be considered in dict
        :type tfidf_threshold: int
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        # todo: include threshold for tfidf value instead of n_words parameter

        start = time.time()

        # todo: temp Remove stopwords
        #nltk.download('stopwords')
        #nltk.download('punkt')
        german_stop_words = stopwords.words('german')

        def custom_tokenizer(text):

            text_tok = word_tokenize(text) #Tokenize
            text_tok_sw = [word for word in text_tok if not word in german_stop_words] #Remove stopwords
            text_tok_sw_alphanum = [word for word in text_tok_sw if word.isalnum()] #Remove punctuation
            return text_tok_sw_alphanum

        #df['text_prep'] = df['text_prep'].apply(lambda x: word_tokenize(x))
        #df['text_prep'] = df['text_prep'].apply(lambda x: [word for word in x if not word in german_stop_words])

        # Define vectorizer
        vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)

        # Fit vectorizer on whole corpus
        vectorizer.fit(df['text_prep'])

        # CALCULATE TF-IDF SCORES OF POP CLASSIFIED DOCS
        df_pop = df.loc[df['POPULIST'] == 1]
        # Transform subcorpus labelled as POP
        tfidf_pop_vector = vectorizer.transform(df_pop['text_prep']).toarray()

        # Map tf-idf scores to words in the vocab with separate column for each doc
        wordlist = pd.DataFrame({'term': vectorizer.get_feature_names()})

        #list = pd.DataFrame()
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

        end = time.time()
        print(end - start)
        print('finished tf-idf dict generation')

        return tfidf_dict


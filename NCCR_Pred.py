import os
import glob
import spacy
import re
import time

import pandas as pd
import numpy as np

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

    def generate_tfidf_dict(self, df: pd.DataFrame):
        """
        Preprocess text of corpus
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        start = time.time()
        
        # Define vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Fit vectorizer
        tfidf_vectorizer.fit(df['text'])

        ### CALCULATE TF-IDF SCORES OF POP CLASSIFIED DOCS
        # Transform subcorpus labelled as POP
        df_prep_pop = df.loc[df['POPULIST'] == 1]

        train_tf_idf_pop = tfidf_vectorizer.transform(df_prep_pop['text'])

        feature_array = np.array(tfidf_vectorizer.get_feature_names())
        tfidf_sorting = np.argsort(train_tf_idf_pop.toarray()).flatten()[::-1]

        n = 100
        top_n = feature_array[tfidf_sorting][:n]
        print(top_n)

        end = time.time()
        print(end - start)
        print('finished tf-idf dict generation')

        #data = pd.DataFrame()
        #  # get the first vector out (for the first document)
        # for doc in range(len(train_tf_idf_pop)):
        #     # place tf-idf values in a pandas data frame
        #     df = pd.DataFrame(train_tf_idf_pop[doc].T, index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
        #     data = data.append(df)
        #
        # data = data.sort_values(by=["tfidf"], ascending=False)
        # print(data)



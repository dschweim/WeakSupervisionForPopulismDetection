import os
import glob
import spacy
import de_core_news_sm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from spacy import displacy
from collections import Counter


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

    def generate_labelled_NCCR_corpus(self):
        """
        Generate labelled NCCR corpus by merging txt-fulltexts with their corresponding labels on text-level and save as csv
        :return:
        :rtype:
        """

        # Get filenames of texts
        os.chdir(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Texts\\Texts')
        files = [i for i in glob.glob("*.txt")]

        print(len(files))

        # Remove English files
        #files = [x for x in files if not x.startswith('uk')]

        # Create empty dataframe for texts
        df = pd.DataFrame()

        # Read every txt file in dir and add to df
        for file in files:
            tmp = open(file, encoding="ISO-8859-1").read()
            txt = pd.DataFrame({'ID': [file],
                                'text': [tmp]})
            df = df.append(txt)

        # Import corpus with populism labels
        table_text = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Text_Table.txt', delimiter="\t", encoding="ISO-8859-1")

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

        return df_combined_de


    def preprocess_corpus(self, corpus):
        """
        Preprocess text of corpus
        :param corpus: Dataset to preprocess
        :type corpus:  DataFrame
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        # 1. Remove characters

        # tbd regex
        corpus_prep = corpus

        # 2. Tokenization

        # 3. Stemming

        # 4. Tagging


        # # todo
        # # Extract texts from corpus
        # texts = [content for content in corpus["text"]]
        #
        # import random
        # random.choices(texts, k=10)
        #
        # # Load corpus
        # nlp = spacy.load("de_core_news_sm")
        #
        # doc = nlp(corpus[corpus.ID == "au_pm_el_02_1001.txt"])
        #
        # print(doc)
        #
        #
        # # Sentiment
        # return corpus_prep


    def generate_train_test_split(self, corpus):
        """
        Generate train test split
        :return:
        :rtype:
        """

        train, test = train_test_split(corpus, test_size=0.2, random_state=0)

        return train, test


    # missings = table_text[~table_text.ID.isin(files)]
    # missings_list = [missings]
    # print(missings_list)

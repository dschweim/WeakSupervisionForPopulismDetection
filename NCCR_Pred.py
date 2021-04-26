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

        # Remove remaining duplicates




        # Save created German corpus
        df_combined_de.to_csv(f'{self.output_path}\\labelled_nccr_corpus_DE.csv', index=True)

        # todo: Remove duplicates from list
        duplicates = df_combined_de[df_combined_de.duplicated(subset=['ID'], keep=False)]


        not_sample = df_combined_de.loc[df_combined_de['Bemerkungen'] == 'Does not belong to the sample / ']



        # Merge combined df with full_speaker, full_target, full_issue
        full_speaker = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Fulltext_Speaker.csv')
        full_issue = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Fulltext_Issue.csv')
        full_target = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Fulltext_Target.csv')

        table_text_combined = pd.merge(df_combined_de, full_speaker, on='ID', how='outer')
        table_text_combined.rename(columns={"Unit_ID": "Unit_ID_SPK", "Spr_ID": "Spr_ID_SPK",
                                            "Wording": "Wording_SPK", "Fulltext": "Fulltext_SPK"}, inplace=True)
        table_text_combined = pd.merge(table_text_combined, full_issue, on='ID', how='outer')
        table_text_combined.rename(columns={"Unit_ID": "Unit_ID_ISS", "Unit_ID01": "Unit_ID01_ISS",
                                            "Spr_ID": "Spr_ID_ISS", "Auto_Coding": "Auto_Coding_ISS",
                                            "Wording": "Wording_ISS", "Fulltext": "Fulltext_ISS"}, inplace=True)
        table_text_combined = pd.merge(table_text_combined, full_target, on='ID', how='outer', indicator=True)
        table_text_combined.rename(columns={"Unit_ID": "Unit_ID_TGT", "Unit_ID01": "Unit_ID01_TGT",
                                            "Spr_ID": "Spr_ID_TGT", "Tgt_ID": "Tgt_ID_TGT",
                                            "Wording": "Wording_TGT", "Fulltext": "Fulltext_TGT"}, inplace=True)

        # Remove rows with "UK" ID
        table_text_combined_de = table_text_combined[~table_text_combined['ID'].astype(str).str.startswith('uk')]

        # todo: How to handle multiple keys

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

        start = time.time()

        #nltk.download('stopwords')
        #nltk.download('punkt')
        german_stop_words = stopwords.words('german')
        german_stop_words.append('fuer')

        def custom_tokenizer(text):
            text_tok = word_tokenize(text) #Tokenize
            text_tok_sw = [word for word in text_tok if not word in german_stop_words] #Remove stopwords
            text_tok_sw_alphanum = [word for word in text_tok_sw if word.isalnum()] #Remove punctuation
            return text_tok_sw_alphanum

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


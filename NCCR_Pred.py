import os
import glob
import spacy

import pandas as pd

from sklearn.model_selection import train_test_split
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

        df.to_csv(f'{self.output_path}\\concatenated_texts.csv', index=True)

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

        return df_combined_de

    def preprocess_corpus(self, corpus):
        """
        Preprocess text of corpus
        :param corpus: Dataset to preprocess
        :type corpus:  DataFrame
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        ### CALCULATE TF-IDF SCORES OF POP CLASSIFIED DOCS

        # # todo: Remove characters at beginning of texts using regex

        # Load corpus
        #nlp = spacy.load("de_core_news_sm")

        # Pre-process corpus
        #docs = list(nlp.pipe(corpus['text']))
        #print(docs)
        #corpus['doc'] = [nlp(text) for text in corpus.text]
        #print(corpus.sample(3))

        # # Sentiment
        # return corpus_prep


        # Calculate tf-idf scores
        vectorizer = TfidfVectorizer()

        # todo: separate handling of train/test
        train_data = corpus.loc[corpus['POPULIST'] == 1]

        train_tf_idf = vectorizer.fit_transform(train_data['text']).toarray()

        data = pd.DataFrame()
         # get the first vector out (for the first document)
        for doc in range(len(train_tf_idf)):
            # place tf-idf values in a pandas data frame
            df = pd.DataFrame(train_tf_idf[doc].T, index=vectorizer.get_feature_names(), columns=["tfidf"])
            data = data.append(df)

        data = data.sort_values(by=["tfidf"], ascending=False)
        print(data)

    def generate_train_test_split(self, corpus):
        """
        Generate train test split
        :return:
        :rtype:
        """

        train, test = train_test_split(corpus, test_size=0.2, random_state=42)

        return train, test

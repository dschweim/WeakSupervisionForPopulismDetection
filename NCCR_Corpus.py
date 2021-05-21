import os
import glob
import re
import time
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from util import standardize_party_naming
pd.options.mode.chained_assignment = None


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
        with their corresponding labels on text-level and join with full_target and target_table
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

        # Join both dataframes (INNER JOIN)
        df_combined = pd.merge(df, table_text, on='ID')

        # Filter on German files
        df_combined_de = df_combined[df_combined.Sample_Lang == 'Deutsch']

        # Find duplicates
        df_combined_de.reset_index(inplace=True)
        duplicates_list = df_combined_de[df_combined_de.duplicated(subset=['ID'], keep=False)]['ID']

        ## Remove duplicates
        # Remove duplicates based on Bemerkung
        df_drop = df_combined_de.loc[((df_combined_de['Bemerkungen'] == 'Does not belong to the sample') |
                                      (df_combined_de['Bemerkungen'] == 'Does not belong to the sample / ') |
                                      (df_combined_de['Bemerkungen'] == 'I already coded this Facebook-Post. / '))
                                     & df_combined_de['ID'].isin(duplicates_list)]

        ## Manually check remaining duplicates
        # Remove duplicate: wrong genre
        df_drop = df_drop.append(df_combined_de.loc[(df_combined_de.ID == 'de_pr_el_13_011004.txt') &
                                                    (df_combined_de.Genre == 26)])

        # Remove duplicate: missing infos
        df_drop = df_drop.append(df_combined_de.loc[(df_combined_de.ID == 'cd_pm_el_11_80016.txt') &
                                                    (df_combined_de.Genre.isnull())])

        # Remove duplicate: wrong main_issue
        df_drop = df_drop.append(df_combined_de.loc[(df_combined_de.ID == 'de_pr_el_13_090015.txt') &
                                                    (df_combined_de.Main_Issue.values == '[\'2100\']')])
        df_drop = df_drop.append(df_combined_de.loc[(df_combined_de.ID == 'de_pr_el_13_061272.txt') &
                                                    (df_combined_de.Main_Issue.values == '[\'2103\']')])
        df_drop = df_drop.append(df_combined_de.loc[(df_combined_de.ID == 'de_pr_el_13_051033.txt') &
                                                    (df_combined_de.Main_Issue.values == '[\'0109\']')])
        df_drop = df_drop.append(df_combined_de.loc[(df_combined_de.ID == 'de_pr_el_13_021047.txt') &
                                                    (df_combined_de.Main_Issue.values == '[\'0199\']')])
        df_drop = df_drop.append(df_combined_de.loc[(df_combined_de.ID == 'de_pm_el_83_50001.txt') &
                                                    (df_combined_de.Main_Issue.values == '[\'0100\', \'0200\']')])

        # Drop duplicates
        df_combined_de = df_combined_de.drop(df_combined_de.index[[df_drop.index.values]])

        ## Join combined_df with full_target and target_table (todo: target_table, full_issue, issue_table)
        # Load dataframes
        full_target = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Fulltext_Target.csv')

        # Drop Duplicates
        full_target.drop_duplicates(inplace=True)

        # Join dfs
        df_combined_de_x_target = pd.merge(df_combined_de, full_target, on='ID')

        # Include source indicator column
        df_combined_de_x_target['Source'] = 'Target'

        # Sort by ID & reset index
        df_combined_de_x_target.sort_values(by='ID', inplace=True)
        df_combined_de_x_target.reset_index(inplace=True)

        # Exclude examples without wording
        df_combined_de_x_target_av = df_combined_de_x_target.dropna(subset=['Wording'])

        # Save created both corpus
        df_combined_de_x_target.to_csv(f'{self.output_path}\\NCCR_combined_corpus_DE_wording_all.csv', index=True)
        df_combined_de_x_target_av.to_csv(f'{self.output_path}\\NCCR_combined_corpus_DE_wording_available.csv',
                                          index=True)

        end = time.time()
        print(end - start)
        print('finished NCCR labelled corpus generation')

        return df_combined_de_x_target, df_combined_de_x_target_av

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

        # Define function to preprocess text column
        def preprocess_text(text):
            # Remove standard text info at beginning of text
            text = re.sub(r'^(\n|.)*--', '', text)

            # Remove linebreaks and extra spaces
            text = " ".join(text.split())

            return text

        # Define function to preprocess wording column
        def preprocess_wording(wording):
            # Replace special characters #todo: simplify replacement
            wording = wording.replace("ae", "ä").replace("ue", "ü").replace("oe", "ö").replace("Oe", "Ö") \
                .replace("Ae", "Ä").replace("Ue", "Ü").replace("/", "").replace("Ausserdem", "Außerdem") \
                .replace("ausserdem", "außerdem").replace("Massnahme", "Maßnahme").replace("<ORD:65430>", "")

            # Remove linebreaks and extra spaces
            wording = " ".join(wording.split())

            return wording

        # Define function to retrieve party from text column
        def retrieve_party(text, sampletype):

            # Press release
            if sampletype == 'PressRelease':
                grp_index = 3
                party = re.search(r'(^(.|\n)*Press Release from Party: )(\w*)', text)

            # Party Manifesto
            elif sampletype == 'PartyMan':
                grp_index = 3
                party = re.search(r'(^(.|\n)*Party Manifesto: )(\w*\s\w*)', text)

            # Past Party Manifesto
            elif sampletype == 'Past_PartyMan':
                grp_index = 3
                party = re.search(r'(^(.|\n)*Party Manifesto, )(\w*)', text)

            # Social Media
            elif sampletype == 'SocialMedia':
                grp_index = 6
                party = re.search(r'(^(.|\n)*Full Name: )(.*, )(.* )(\()(.*)(\))(.*)', text)

            if party is None:
                return None
            else:
                return party.group(grp_index)

        # Retrieve year from date column
        def retrieve_year(date):
            year = re.search(r'^\d\d.\d\d.(\d{4})', date)

            if year is None:
                return None
            else:
                return year.group(1)

        # Apply preprocess_text function to whole text column
        df['text_prep'] = df['text'].apply(lambda x: preprocess_text(x))
        # Apply preprocess_wording function to whole wording column
        df['Wording'] = df['Wording'].apply(lambda x: preprocess_wording(x))

        # Apply retrieve_party function to whole text column depending on sampletype
        df['party'] = df.apply(lambda x: retrieve_party(x['text'], x['Sample_Type']), axis=1)

        # Standardize party naming
        df['party'] = df['party'].apply(lambda x: standardize_party_naming(x))

        # Apply retrieve_year function to whole date column
        df['Date'] = df['Date'].astype(str)
        df['year'] = df['Date'].apply(lambda x: retrieve_year(x))
        df['year'] = df['year'].astype(int)

        # Generate additional column with segments of text that contain relevant content using Wording column
        df_seg = self.__retrieve_segments(df)

        # Save pre-processed corpus
        df_seg.to_csv(f'{self.output_path}\\NCCR_combined_corpus_DE_wording_available_{label}.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished dataset preprocessing for ' + label)

        return df

    @staticmethod
    def __custom_tokenizer(text):
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

    @staticmethod
    def __retrieve_segments(df: pd.DataFrame):
        """
        Retrieve segments from fulltext that correspond to content in Wording column
        :param df: Dataframe for retrieval of segments
        :type df: DataFrame
        :return: Returns Dataframe with Column for Segments
        :rtype:  DataFrame
        """

        # Load Spacy Model and include sentencizer
        nlp = spacy.load("de_core_news_lg", exclude=['tok2vec', 'tagger', 'morphologizer', 'parser',
                                                     'attribute_ruler', 'lemmatizer'])
        nlp.add_pipe("sentencizer")

        # Generate spacy docs
        df['doc'] = list(nlp.pipe(df['text_prep']))

        # Define function to find index of tokens that match Wording content
        def get_matches(doc: spacy.tokens.doc.Doc, wording: str):
            # Define Spacy Matcher
            matcher = PhraseMatcher(nlp.vocab)
            # Add patterns from nlp-preprocessed Wording column
            matcher.add("WORDING", [nlp(wording)])
            # Get matches
            matches = matcher(doc)

            return matches

        df['wording_matches'] = df.apply(lambda x: get_matches(x['doc'], x['Wording']), axis=1)

        # Define function to retrieve sentences that correspond to matched tokens
        def collect_sentences(doc: spacy.tokens.doc.Doc, matches: list, triples: bool):
            # Skip for empty matches
            if matches is None:
                return None
                # todo: return doc
            else:
                # Define empty list/string
                sentences = []
                sentence_triples = ''

                # Retrieve main sentence + pre- and succeeding sentence of each match
                for match_id, start, end in matches:
                    # Retrieve main sentence
                    main_sent = doc[start:end].sent.text
                    # Append to list
                    sentences.append([main_sent])

                    if triples:
                        # check if main_sent is first sentence,  if so return empty string,
                        # otherwise return previous sentence
                        if doc[start].sent.start - 1 < 0:
                            previous_sent = ''
                        else:
                            previous_sent = doc[doc[start].sent.start - 1].sent.text + ' '

                        # check if main_sent is last sentence, if so return empty string,
                        # otherwise return following sentence
                        if doc[start].sent.end + 1 >= len(doc):
                            following_sent = ''
                        else:
                            following_sent = ' ' + doc[doc[start].sent.end + 1].sent.text

                        # Append triples to string
                        sentence_triples = sentence_triples + previous_sent + main_sent + following_sent

                if triples:
                    return sentence_triples
                else:
                    return sentences

        # Run function to retrieve main sentence and sentence triples
        df['wording_sentence'] = \
            df.apply(lambda x: collect_sentences(x['doc'], x['wording_matches'], triples=False), axis=1)
        df['wording_sentence_triples'] = \
            df.apply(lambda x: collect_sentences(x['doc'], x['wording_matches'], triples=True), axis=1)

        # todo: handle Columns with "Wording" Non-match:
        non = df[~df['wording_matches'].astype(bool)]

        return df

    def generate_tfidf_dict(self, df: pd.DataFrame, n_words: int):
        """
        Calculate tf-idf scores of docs and return top n words with tfidf above threshold
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param n_words: Number of words to be included
        :type n_words: float
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        start = time.time()

        # Define vectorizer
        vectorizer = TfidfVectorizer(tokenizer=self.__custom_tokenizer)

        # Fit vectorizer on whole corpus
        vectorizer.fit(df['Wording'])

        # CALCULATE TF-IDF SCORES OF POP CLASSIFIED DOCS
        df_pop = df.loc[df['POPULIST'] == 1]
        # Transform subcorpus labelled as POP
        tfidf_pop_vector = vectorizer.transform(df_pop['Wording']).toarray()

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
        Calculate tf-idf scores of docs per country and return top n words with tfidf above threshold
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param n_words: Number of words to be included
        :type n_words: float
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
            vectorizer.fit(df_country['Wording'])

            # CALCULATE TF-IDF SCORES OF POP CLASSIFIED DOCS
            df_country_pop = df_country.loc[df_country['POPULIST'] == 1]

            # Transform subcorpus labelled as POP
            tfidf_pop_vector = vectorizer.transform(df_country_pop['Wording']).toarray()

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
        Calculate tf-idf scores of docs and return top n words with tfidf above threshold
        :param df: Trainset from which to construct dict
        :type df:  DataFrame
        :param n_words: Number of words to be included
        :type n_words: float
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
        content_pop = ' '.join(df_pop["Wording"])
        content_nonpop = ' '.join(df_nonpop["Wording"])

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

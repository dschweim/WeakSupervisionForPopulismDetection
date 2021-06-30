import os
import glob
import re
import time
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc
import pandas as pd
from util import standardize_party_naming

pd.options.mode.chained_assignment = None


class NCCR_Dataset:
    def __init__(
            self,
            data_path: str,
            output_path: str
    ):
        """
        Class to create and pre-process the NCCR data
        :param data_path: path to data input
        :type data_path: str
        :param data_path: path to data output
        :type data_path: str
        """

        self.data_path = data_path
        self.output_path = output_path
        self.nlp_sent = spacy.load("de_core_news_lg", exclude=['tok2vec', 'tagger', 'morphologizer', 'parser',
                                                               'attribute_ruler', 'lemmatizer'])
        self.nlp_sent.add_pipe("sentencizer")

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

        ## Join combined_df with full_target and target_table
        # Load dataframes
        full_target = pd.read_csv(f'{self.data_path}\\NCCR_Content\\NCCR_Content\\Fulltext_Target.csv')

        # Drop Duplicates
        full_target.drop_duplicates(inplace=True)

        # Join dfs
        df_combined_de_x_target = pd.merge(df_combined_de, full_target, on='ID')

        # Remove duplicate: wrong speaker
        df_drop = df_combined_de_x_target.loc[(df_combined_de_x_target.ID == 'au_pr_el_13_010316.txt') &
                                              (df_combined_de_x_target.Spr_ID == 11801)]

        df_combined_de_x_target = df_combined_de_x_target.drop(df_combined_de_x_target.index[[df_drop.index.values]])

        # Include source indicator column
        df_combined_de_x_target['Source'] = 'Target'

        # Sort by ID & reset index
        df_combined_de_x_target.sort_values(by='ID', inplace=True)
        df_combined_de_x_target.reset_index(inplace=True)

        # Exclude examples without wording
        df_combined_de_x_target_av = df_combined_de_x_target.dropna(subset=['Wording'])

        # Save created corpus
        df_combined_de_x_target_av.to_csv(f'{self.output_path}\\NCCR_combined_corpus_DE_wording_available.csv',
                                          index=True)

        end = time.time()
        print(end - start)
        print('finished NCCR labelled corpus generation')

        return df_combined_de_x_target_av

    def preprocess_corpus(self, df: pd.DataFrame):
        """
        Preprocess text of corpus
        :param df: Dataset to preprocess
        :type df:  DataFrame
        :return: Returns preprocessed Dataset
        :rtype:  DataFrame
        """

        start = time.time()

        # Apply preprocess_text function to whole text column
        df['text_prep'] = df['text'].apply(lambda x: self.__remove_intro(x))

        # Apply retrieve_party function to whole text column depending on sampletype
        df['party'] = df.apply(lambda x: self.__retrieve_party(x['text'], x['Sample_Type']), axis=1)

        # Standardize party naming
        df['party'] = df['party'].apply(lambda x: standardize_party_naming(x))

        # Apply retrieve_year function to whole date column
        df['Date'] = df['Date'].astype(str)
        df['year'] = df['Date'].apply(lambda x: self.__retrieve_year(x))
        df['year'] = df['year'].astype(int)

        # Generate additional column with segments of text that contain relevant content using Wording column
        df_seg = self.__retrieve_segments(df)

        # Save pre-processed corpus
        df_seg.to_csv(f'{self.output_path}\\NCCR_combined_corpus_DE_wording_available_prep.csv', index=False)

        end = time.time()
        print(end - start)
        print('finished dataset preprocessing')

        return df_seg

    @staticmethod
    def __remove_intro(text: str):
        """
        Retrieve standard intro from text
        :param text: String of text
        :type date: str
        :return: Returns subtext without intro
        :rtype:  str
        """
        # Define function to preprocess text column

        # Remove standard text info at beginning of text
        text = re.sub(r'^(\n|.)*--', '', text)

        # Remove linebreaks and extra spaces
        text = " ".join(text.split())

        return text

    @staticmethod
    def __retrieve_year(date: str):
        """
        Retrieve year from date
        :param date: String of date
        :type date: str
        :return: Returns year of date
        :rtype:  str
        """

        # Retrieve year from date column
        year = re.search(r'^\d\d.\d\d.(\d{4})', date)

        if year is None:
            return None
        else:
            return year.group(1)

    @staticmethod
    def __retrieve_party(text: str, sampletype: str):
        """
        Retrieve party from text
        :param text: String of text
        :type text: str
        :param sampletype: Indicator of sample type
        :type sampletype: str
        :return: Returns party name
        :rtype:  str
        """

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

    def __retrieve_segments(self, df: pd.DataFrame):
        """
        Retrieve segments from fulltext that correspond to content in Wording column
        :param df: Dataframe for retrieval of segments
        :type df: DataFrame
        :return: Returns Dataframe with Column for Segments
        :rtype:  DataFrame
        """

        # Apply temporary textual preprocessing function to text and wording column
        df['text_temp'] = df['text_prep'].apply(lambda x: self.__standardize_text(x))
        df['Wording_temp'] = df['Wording'].apply(lambda x: self.__standardize_text(x))

        # Generate spacy docs
        df['doc_temp'] = list(self.nlp_sent.pipe(df['text_temp']))
        df['Wording_doc_temp'] = list(self.nlp_sent.pipe(df['Wording_temp']))

        ## FIX SPELLING
        # Define tokens to fix in Wording and Text
        replacement_dict = {"Parteienfamilie": [{'LOWER': 'parteienfamili'}], "Volkspartei": [{'LOWER': 'volksparte'}],
                            "Spindelegger": [{'LOWER': 'spindelegge'}], "Kanzler": [{'LOWER': 'anzler'}],
                            "sozialistischen": [{'LOWER': 'ozialistischen'}], "Ein": [{'LOWER': '.ein'}],
                            "Josef": [{'LOWER': 'osef'}], "das": [{'LOWER': 'as'}], "Nun": [{'LOWER': 'un'}],
                            "Anstatt": [{'LOWER': 'nstatt'}], "Ich": [{'LOWER': r'\"ich'}],
                            "Danke": [{'LOWER': r'\"danke'}], "LINKE": [{'LOWER': "link"}],
                            "FDP-Bundestagsfraktion": [{'LOWER': "fdp-"}], "CDUCSU-Fraktion": [{'LOWER': "cducsu-"}],
                            "Den": [{'LOWER': "n"}], "hat": [{'LOWER': "h"}], "ist": [{'LOWER': "i"}],
                            "Der": [{'LOWER': "r"}], "Laender": [{'LOWER': "l"}], "Die": [{'LOWER': "-die"}],
                            "Viel": [{'LOWER': "-viel"}], "einerseits": [{'LOWER': "einerseit"}],
                            "Eine": [{'LOWER': "-eine"}],
                            'mussten': [{'TEXT': "mußten"}], 'veranlasst': [{'TEXT': "veranlaßt"}],
                            'Einfluss': [{'TEXT': "Einfluß"}], 'müsste': [{'TEXT': "müßte"}],
                            'Abrissbirne': [{'TEXT': "Abrißbirne"}], 'Datenmissbrauch': [{'TEXT': "Datenmißbrauch"}],
                            'Hasspredigers': [{'TEXT': "Haßpredigers"}], 'missbraucht': [{'TEXT': "mißbraucht"}],
                            'verantwortungsbewusste': [{'TEXT': "verantwortungsbewußte"}],
                            'Schlussstrich': [{'TEXT': "Schlußstrich"}], 'veranlassten': [{'TEXT': "veranlaßten"}],
                            'Genuss': [{'TEXT': "Genuß"}], 'Verlässlichkeit': [{'TEXT': "Verläßlichkeit"}],
                            'Verlaesslichkeit': [{'TEXT': "Verlaeßlichkeit"}], 'dass': [{'TEXT': "daß"}],
                            'passt': [{'TEXT': "paßt"}], 'musste': [{'TEXT': "mußte"}],
                            'Asylmissbrauch': [{'TEXT': "Asylmißbrauch"}], 'missbrauchen': [{'TEXT': "mißbrauchen"}],
                            'Misslage': [{'TEXT': "Mißlage"}], 'Dass': [{'TEXT': "Daß"}],
                            'Kindesmissbrauch': [{'TEXT': "Kindesmißbrauch"}],
                            'Ausschlusskriterium': [{'TEXT': "Ausschlußkriterium"}], 'lässt': [{'TEXT': "läßt"}],
                            'laesst': [{'TEXT': "laeßt"}], 'Biss': [{'TEXT': "Biß"}],
                            'Missbrauch': [{'TEXT': "Mißbrauch"}], 'unmissverständlich': [{'TEXT': "unmißverständlich"}],
                            'unmissverstaendlich': [{'TEXT': "unmißverstaendlich"}], 'Schloss': [{'TEXT': "Schloß"}],
                            'befasste': [{'TEXT': "befaßte"}], 'Ausschluss': [{'TEXT': "Ausschluß"}],
                            'Bewusstsein': [{'TEXT': "Bewußtsein"}], 'Schluss': [{'TEXT': "Schluß"}],
                            'machtbewussten': [{'TEXT': "machtbewußten"}],
                            'Ernährungsbewusstsein': [{'TEXT': "Ernährungsbewußtsein"}], 'Ausschuss': [{'TEXT': "Ausschuß"}],
                            'Abfluss': [{'TEXT': "Abfluß"}], 'verhasst': [{'TEXT': "verhaßt"}],
                            'müsst': [{'TEXT': "müßt"}], 'muesst': [{'TEXT': "mueßt"}],
                            'Parteiausschlussverfahren': [{'TEXT': "Parteiausschlußverfahren"}],
                            'Unermessliche': [{'TEXT': "Unermeßliche"}], 'Hassprediger': [{'TEXT': "Haßprediger"}],
                            'muss': [{'TEXT': "muß"}], 'bedauert': [{'TEXT': "bedäurt"}],
                            'Steuern': [{'TEXT': "Steürn"}], 'Steuerzahler': [{'TEXT': "Steürzahler"}],
                            'Steuerzahlergedenktag': [{'TEXT': "Steürzahlergedenktag"}],
                            'Steuer-': [{'TEXT': "Steür-"}],  'Steuersystem': [{'TEXT': "Steürsystem"}],
                            'Steuerlast': [{'TEXT': "Steürlast"}], 'Steuereinnahmen': [{'TEXT': "Steüreinnahmen"}],
                            'Lohnsteuerpflichtigen': [{'TEXT': "Lohnsteürpflichtigen"}],
                            'anlässlich': [{'TEXT': "anläßlich"}],
                            'Vermögenssteuer': [{'TEXT': "  'Vermögenssteür"}], 'Einschluss': [{'TEXT': "Einschluß"}]}

        # Replace each token in dict with it's corresponding corrected replacement
        for key in replacement_dict:
            df['Wording_doc_temp'] = \
                df['Wording_doc_temp'].apply(lambda x: self.__fix_spelling(x, key, replacement_dict[key]))
            df['doc_temp'] = \
                df['doc_temp'].apply(lambda x: self.__fix_spelling(x, key, replacement_dict[key]))

        ## FIX UMLAUTS
        # Define umlauts
        umlauts_dict = {"ä": [{"TEXT": {"REGEX": "ä"}}],
                        "Ä": [{"TEXT": {"REGEX": "Ä"}}],
                        "ö": [{"TEXT": {"REGEX": "ö"}}],
                        "Ö": [{"TEXT": {"REGEX": "Ö"}}],
                        "ü": [{"TEXT": {"REGEX": "ü"}}],
                        "Ü": [{"TEXT": {"REGEX": "Ü"}}],
                        "ß": [{"TEXT": {"REGEX": "ß"}}]}

        # Create empty list for tokens to replace
        replacement_list_out = []

        # Iterate over umlauts dict
        for key in umlauts_dict:

            # Define current matching pattern
            pattern = umlauts_dict[key]

            # Define Matcher and add current pattern
            token_matcher = Matcher(self.nlp_sent.vocab)
            token_matcher.add("REPLACE", [pattern])

            # Create empty list for tokens in current iteration
            current_replacement_list = []

            # Iterate over docs
            for index, row in df.iterrows():
                # Get matches of tokens which contain current umlaut
                matches = token_matcher(row.doc_temp)

                # For each match, do:
                for match_id, start, end in matches:
                    # Get the token text
                    token = row.doc_temp[start:end].text
                    # Add token to replacement list
                    current_replacement_list.append(token)

            # Add list of unique tokens to replacement list
            replacement_list_out.extend(list(set(current_replacement_list)))

        # Create similar list of tokens with alternative spelling
        replacement_list_in = replacement_list_out.copy()
        for index, token in enumerate(replacement_list_in):
            token_fixed = token.replace("ä", "ae").replace("Ä", "Ae").replace("ö", "oe") \
                .replace("Ö", "Oe").replace("ü", "ue").replace("Ü", "Ue").replace("ß", "ss")
            replacement_list_in[index] = token_fixed

        # Generate dictionary for replacement
        keys = replacement_list_out
        values = replacement_list_in
        umlauts_token_dict = dict(zip(keys, values))

        # Replace tokens with umlauts according to umlauts_token_dict in Text and Wording
        df['doc_temp'] = df['doc_temp'].astype(str)
        df['Wording_doc_temp'] = df['Wording_doc_temp'].astype(str)

        for key in umlauts_token_dict:
            df['doc_temp'] = \
                df['doc_temp'].apply(lambda x: self.__standardize_umlauts(x, key, umlauts_token_dict[key]))
            df['Wording_doc_temp'] = \
                df['Wording_doc_temp'].apply(lambda x: self.__standardize_umlauts(x, key, umlauts_token_dict[key]))

        # Generate spacy docs
        df['doc_temp'] = list(self.nlp_sent.pipe(df['doc_temp']))
        df['Wording_doc_temp'] = list(self.nlp_sent.pipe(df['Wording_doc_temp']))

        # Retrieve Wording-Text-matches
        df['wording_matches'] = df.apply(lambda x: self.__get_matches(x['doc_temp'], x['Wording_doc_temp']), axis=1)

        # Run function to retrieve main sentence and sentence triples
        df['wording_sentence'] = \
            df.apply(lambda x: self.__collect_sentences(x['doc_temp'], x['wording_matches'], triples=False), axis=1)
        df['wording_segments'] = \
            df.apply(lambda x: self.__collect_sentences(x['doc_temp'], x['wording_matches'], triples=True), axis=1)

        # Calculate number of matches
        df['match_count'] = df['wording_matches'].apply(lambda x: len(x))

        # Retrieve count of sentences in Wording
        df['Wording_sent_count'] = df['Wording_doc_temp'].apply(lambda x: len(list(x.sents)))

        # Drop examples where sent_count > 3
        df = df.loc[df.Wording_sent_count <= 3]

        # Where Wording_sent_count is 2 or 3, replace Wording segment with Wording and set match_count to -1
        df.loc[df.Wording_sent_count == 2, 'wording_segments'] =\
            df.loc[df.Wording_sent_count == 2].Wording_doc_temp.astype(str)
        df.loc[df.Wording_sent_count == 2, 'match_count'] = -1

        df.loc[df.Wording_sent_count == 3, 'wording_segments'] =\
            df.loc[df.Wording_sent_count == 3].Wording_doc_temp.astype(str)
        df.loc[df.Wording_sent_count == 3, 'match_count'] = -1

        ## NO MATCH
        # Retrieve corpus with no match for manual fixing
        df_none = df.loc[df.match_count == 0]

        # Retry to retrieve match using only n first tokens of Wording
        df_none['Wording_doc_temp'] = df_none['Wording_doc_temp'].apply(
            lambda x: self.__get_sub_wording(x, n_tokens=10))

        # Retrieve Wording-Text-matches
        df_none['wording_matches'] = df_none.apply(lambda x: self.__get_matches(x['doc_temp'], x['Wording_doc_temp']),
                                                   axis=1)

        # Run function to retrieve main sentence and sentence triples
        df_none['wording_sentence'] = \
            df_none.apply(lambda x: self.__collect_sentences(x['doc_temp'], x['wording_matches'], triples=False),
                          axis=1)
        df_none['wording_segments'] = \
            df_none.apply(lambda x: self.__collect_sentences(x['doc_temp'], x['wording_matches'], triples=True), axis=1)

        # Calculate number of matches
        df_none['match_count'] = df_none['wording_matches'].apply(lambda x: len(x))

        # Split dataset between subwording retrieved matches and remaining none_matches
        df_none_submatched = df_none.loc[df_none.match_count == 1]
        df_none = df_none.loc[df_none.match_count == 0]

        # Generate additional columns
        df_none['doc_tokens'] = df_none['doc_temp'].apply(lambda x: [token.text for token in x])
        df_none['wording_tokens'] = df_none['Wording_doc_temp'].apply(lambda x: [token.text for token in x])

        # Save corpus
        df_none.drop(columns=['level_0', 'index'], inplace=True)
        df_none.reset_index(inplace=True)
        df_none.to_csv(f'{self.output_path}\\df_none_match.csv')

        ## MANUAL REPLACEMENT NONE_MATCH
        # Only keep rows with 1 and -1 match for main corpus
        df = df.loc[(df.match_count == 1) | (df.match_count == -1)]

        # Add subwording-based matches to corpus
        df = df.append(df_none_submatched)

        # Replace Wording for corpus with no match using manual_replacement_table
        replace_table_non = pd.read_csv(f'{self.output_path}\\manual_replacement\\none_match_replace_table.csv')

        # Only keep rows for which replacement is available
        df_none = pd.merge(df_none, replace_table_non, left_on='index', right_on='ID_non')

        # Generate spacy doc
        df_none['Wording_doc_temp'] = list(self.nlp_sent.pipe(df_none['Wording_fixed']))

        # Retrieve Wording-Text-matches
        df_none['wording_matches'] = df_none.apply(lambda x: self.__get_matches(x['doc_temp'], x['Wording_doc_temp']),
                                                   axis=1)

        # Run function to retrieve main sentence and sentence triples
        df_none['wording_sentence'] = \
            df_none.apply(lambda x: self.__collect_sentences(x['doc_temp'], x['wording_matches'], triples=False),
                          axis=1)
        df_none['wording_segments'] = \
            df_none.apply(lambda x: self.__collect_sentences(x['doc_temp'], x['wording_matches'], triples=True), axis=1)

        # Calculate match_count for manually verified examples
        df_none['match_count'] = df_none['wording_matches'].apply(lambda x: len(x))

        # Add manually fixed matches to main corpus
        df = df.append(df_none)

        # Only keep rows with 1 and -1 match for main corpus
        df = df.loc[(df.match_count == 1) | (df.match_count == -1)]

        # Drop redundant columns
        df.drop(columns=['level_0', 'index'], inplace=True)
        # Delete temp columns
        df.drop(columns=
                ['text_temp', 'Wording_temp', 'doc_temp', 'Wording_doc_temp', 'doc_tokens', 'wording_tokens',
                 'Wording_fixed'], inplace=True)

        print('GENERATED CORPUS: ' + str(len(df)) + ' EXAMPLES')

        return df

    @staticmethod
    def __standardize_text(text: str):
        """
        Temporary preprocessing for textual content
        :param text: Text to standardize
        :type text: str
        :return: Returns standardized text
        :rtype:  str
        """

        # Replace special characters
        text = text.replace("/", "").replace("@", "").replace("#", "") \
            .replace(r"\\x84", "").replace(r"\\x93", "") \
            .replace(r"\x84", "").replace(r"\x93", "").replace(r"\x96", "") \
            .replace(r"\\x96", "").replace("t.coIXcqTPZHsM+", "") \
            .replace("<ORD:65427>", "").replace("Elite", "Elite").replace(r"\"Begabung\"", "Begabung") \
            .replace("Begabung", "Begabung").replace("funktioniert", "funktioniert") \
            .replace("Nächstenliebe", "Nächstenliebe").replace("Wir", "Wir") \
            .replace("Arbeiterparteien", "Arbeiterparteien") \
            .replace("", "") \
            .replace("<ORD:65412>", "").replace("<ORD:65430>", "").replace("<quot>", r"\"") \
            .replace("<ORD:65440>", "").replace("<ORD:65451>", "").replace("<TAB>", "") \
            .replace("F.D.P.", "FDP").replace(".dieLinke", "dieLinke") \
            .replace("é", "e").replace("è", "e").replace("É", "E").replace("È", "E") \
            .replace("à", "a").replace("á", "a").replace("Á", "A").replace("À", "A") \
            .replace("ò", "o").replace("ó", "o").replace("Ó", "O").replace("Ò", "O") \
            .replace("ç", "c").replace(r"\\", "")
        text = " ".join(text.split())  # Remove additional whitespaces

        return text

    @staticmethod
    def __standardize_umlauts(text: str, replacement: str, token: str):
        """
        correct redundant characters and typos in wording column
        :param text: textual content to correct
        :type text: str
        :param replacement: token to use as replacement
        :type replacement: str
        :param token: token to be replaced
        :type token: str
        :return: Returns corrected textual content as str
        :rtype: str
        """

        # Todo: chck wether subwording replaced (e.g. Ruß + Russlandreisen -> Rußlandreisen (ID: de_pr_el_13_060133.txt)
        text = text.replace(token, replacement)

        return text

    def __fix_spelling(self, doc: spacy.tokens.doc.Doc, replacement: str, pattern: list):
        """
        correct redundant characters, typos and wrong spelling in doc
        :param doc: Wording content to correct
        :type doc: spacy.tokens.doc.Doc
        :param replacement: token to use for replacement
        :type replacement: str
        :param pattern: pattern to use for Matching
        :type pattern: list
        :return: Returns corrected content as doc
        :rtype:  spacy.tokens.doc.Doc
        """

        # Define Matcher
        token_matcher = Matcher(self.nlp_sent.vocab)
        token_matcher.add("REPLACE", [pattern])

        # If no match, return wording as is
        if not token_matcher(doc):
            return doc
        # If match, replace match by corresponding replacement token
        else:
            text = ''
            buffer_start = 0
            for _, match_start, _ in token_matcher(doc):
                if match_start > buffer_start:  # Add skipped token
                    text += doc[buffer_start: match_start].text + doc[match_start - 1].whitespace_
                text += replacement + doc[
                    match_start].whitespace_  # Replace token, with trailing whitespace if available
                buffer_start = match_start + 1
            text += doc[buffer_start:].text

            return self.nlp_sent.make_doc(text)

    def __get_matches(self, doc: spacy.tokens.doc.Doc, wording: spacy.tokens.doc.Doc):
        """
        Find index of tokens that match Wording content
        :param doc: text in which to look for matches
        :type doc: spacy.tokens.doc.Doc
        :param wording: Wording content to look for
        :type wording: str
        :return: returns list of matches
        :rtype:  list
        """

        # Define Spacy Matcher
        matcher = PhraseMatcher(self.nlp_sent.vocab)
        # Add patterns from nlp-preprocessed Wording column
        matcher.add("WORDING", [wording])
        # Get matches
        matches = matcher(doc)

        return matches

    @staticmethod
    def __collect_sentences(doc: spacy.tokens.doc.Doc, matches: list, triples: bool):
        """
        Retrieve sentences that correspond to matched tokens
        :param doc: text from which to retrieve matches
        :type doc: spacy.tokens.doc.Doc
        :param matches: list with index of matches
        :type matches: list
        :param triples: indicator whether to retrieve sentences or triples of sentences
        :type triples: bool
        :return: returns sentences or sentence triples
        :rtype: list
        """

        # Skip for empty matches
        if matches is None:
            return None
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

    @staticmethod
    def __get_sub_wording(wording: spacy.tokens.doc.Doc, n_tokens: int):
        """
        Retireve first n tokens of Wording
        :param wording: Full Wording content
        :type wording: spacy.tokens.doc.Doc
        :param n_tokens: Number of tokens to retrieve
        :type n_tokens: int
        :return: Returns sub wording as doc
        :rtype:  spacy.tokens.doc.Doc
        """

        doc_sub = Doc(wording.vocab, words=[t.text for i, t in enumerate(wording) if i < n_tokens])
        return doc_sub

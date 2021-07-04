import pandas as pd
import spacy
import time
import os
import glob
import xml.etree.ElementTree as ET

pd.options.mode.chained_assignment = None


class BT_Dataset:
    def __init__(
            self,
            data_path: str,
            output_path: str,
            spacy_model: str
    ):
        """
        Class to create and pre-process the Bundestag opendata
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
        self.nlp_sent = spacy.load(spacy_model, exclude=['tagger', 'morphologizer', 'parser',
                                                         'attribute_ruler', 'lemmatizer'])
        self.nlp_sent.add_pipe("sentencizer")

    def generate_bt_corpus(self):
        start = time.time()

        os.chdir(f'{self.data_path}\\BT_OpenData\\bundestag')
        xml_files = [i for i in glob.glob("*.xml")]

        # Create empty dataframe for texts
        df = pd.DataFrame()

        # Read every xml file in dir and extract content
        for file in xml_files:
            tree = ET.parse(file)
            root = tree.getroot()

            # Get date
            text_date = root.find('.//kopfdaten/veranstaltungsdaten/datum').attrib.get('date')

            # Extract speeches
            for speech in root.findall('.//rede'):

                # Extract speech_id
                text_id = speech.attrib.get('id')
                text_speech = ''

                # Iterate over paragraphs
                for p in speech.findall('.//p'):

                    # Extract speaker information
                    if p.attrib.get('klasse') == 'redner':
                        for child in p.getchildren():
                            if child.tag == 'redner':

                                spr_title = None
                                spr_firstname = None
                                spr_lastname = None
                                spr_party = None
                                spr_role_full = None
                                spr_role_short = None

                                spr_id = child.attrib.get('id')
                                for grandchild in child.getchildren():
                                    if grandchild.tag == 'name':
                                        for ggchild in grandchild.getchildren():
                                            if ggchild.tag == 'titel':
                                                spr_title = ggchild.text
                                            elif ggchild.tag == 'vorname':
                                                spr_firstname = ggchild.text
                                            elif ggchild.tag == 'nachname':
                                                spr_lastname = ggchild.text
                                            elif ggchild.tag == 'fraktion':
                                                spr_party = ggchild.text
                                            elif ggchild.tag == 'rolle_lang':
                                                spr_role_full = ggchild.text
                                            elif ggchild.tag == 'rolle_kurz':
                                                spr_role_short = ggchild.text

                    # Extract text content
                    elif p.attrib.get('klasse') in ['J', 'J_1', 'O', 'Z']:
                        if p.text is not None:
                            text = p.text
                        else:
                            text = ''

                        if text_speech == '':
                            text_speech = text

                        else:
                            text_speech = text_speech + ' \n ' + text

                df_speech = pd.DataFrame({'text_id': [text_id],
                                          'text_date': [text_date],
                                          'source_file': [file],
                                          'spr_id': [spr_id],
                                          'spr_title': [spr_title],
                                          'spr_firstname': [spr_firstname],
                                          'spr_lastname': [spr_lastname],
                                          'spr_party': [spr_party],
                                          'spr_role_full': [spr_role_full],
                                          'spr_role_short': [spr_role_short],
                                          'text': [text_speech]
                                          })

                df = df.append(df_speech)

        # Drop rows without text
        df.replace("", float("NaN"), inplace=True)
        df.dropna(subset=["text"], inplace=True)

        # Save concatenated texts
        df.to_csv(f'{self.output_path}\\BT_corpus.csv', index=True)


nccr_df = BT_Dataset(data_path='C:\\Users\\dschw\\Documents\\GitHub\\Thesis\\Data',
                     output_path='C:\\Users\\dschw\\Documents\\GitHub\\Thesis\\Output',
                     spacy_model='de_core_news_lg')

nccr_df.generate_bt_corpus()

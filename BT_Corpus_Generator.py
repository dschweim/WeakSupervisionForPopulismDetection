import pandas as pd
import spacy
import time
import os
import glob
import xml.etree.ElementTree as ET
from lxml import etree
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
            tree = etree.parse(file)
            root = tree.getroot()

            # Get date of file
            speech_date = root.find('.//kopfdaten/veranstaltungsdaten/datum').attrib.get('date')

            # Iterate over speeches
            for speech in root.findall('.//rede'):

                # Get id of speech
                speech_id = speech.attrib.get('id')

                # Reset value
                subspeech_id = None

                # Iterate over J_1s (subspeeches)
                for index, subspeech in enumerate(speech.xpath('p[@klasse=\'J_1\']')):

                    # Set index of current subspech
                    subspeech_id = str(index)

                    # Extract speaker element
                    spr = subspeech.xpath('(preceding-sibling::*)[last()]')[0]

                    # Reset values
                    spr_text = ''
                    spr_id = None
                    spr_title = None
                    spr_firstname = None
                    spr_lastname = None
                    spr_name = None
                    spr_party = None
                    spr_role_full = None
                    spr_role_short = None

                    if spr.tag == 'p':
                        for child in spr.getchildren():
                            if child.tag == 'redner':
                                spr_id = child.attrib.get('id')

                                for gchild in child.getchildren():
                                    for ggchild in gchild.getchildren():
                                        if ggchild.tag == 'titel':
                                            spr_title = ggchild.text
                                        elif ggchild.tag == 'vorname':
                                            spr_firstname = ggchild.text
                                        elif ggchild.tag == 'nachname':
                                            spr_lastname = ggchild.text
                                        elif ggchild.tag == 'fraktion':
                                            spr_party = ggchild.text

                    elif spr.tag == 'name':
                        spr_name = spr.text

                    # todo: Generate joined name attribute + try to extract party

                    # Get count for xpath
                    n = str(index + 1)

                    # Extract text nodes that belong to current subspeech
                    spr_text_j1 = subspeech.xpath('parent::*/p[@klasse=\'J_1\'][' + n + ']')[0].text
                    spr_text_p_list = subspeech.xpath(
                        'parent::*/p[@klasse=\'J_1\'][' + n + ']/following-sibling::p[not (@klasse=\'J_1\') and not (@klasse=\'T\') and count(preceding-sibling::p[@klasse=\'J_1\'])=' + n + ']')

                    # Replace list items with their text content
                    for index, node in enumerate(spr_text_p_list):
                        spr_text_p_list[index] = node.text

                    # Remove NaN
                    spr_text_p_list = [val for val in spr_text_p_list if val]

                    # Concatenate text contents
                    if spr_text_p_list is not None:
                        spr_text_p = ' \n '.join(spr_text_p_list)
                    else:
                        spr_text_p = ''

                    # Join J_1 text and following siblings of type p
                    if spr_text_p == '':
                        spr_text = spr_text_j1
                    else:
                        spr_text = spr_text_j1 + ' \n ' + spr_text_p

                    df_speech = pd.DataFrame({'text_id': [speech_id],
                                              'text_subid': [subspeech_id],
                                              'text_date': [speech_date],
                                              'text_source': [file],
                                              'spr_id': [spr_id],
                                              'spr_title': [spr_title],
                                              'spr_firstname': [spr_firstname],
                                              'spr_lastname': [spr_lastname],
                                              'spr_name': [spr_name],
                                              'spr_party': [spr_party],
                                              'spr_role_full': [spr_role_full],
                                              'spr_role_short': [spr_role_short],
                                              'spr_text': [spr_text]
                                              })

                    df = df.append(df_speech)


        # Drop rows without text
        # df.replace("", float("NaN"), inplace=True)
        # df.dropna(subset=["text"], inplace=True)

        # Save concatenated texts
        df.to_csv(f'{self.output_path}\\BT_corpus.csv', index=True)


nccr_df = BT_Dataset(data_path='C:\\Users\\dschw\\Documents\\GitHub\\Thesis\\Data',
                     output_path='C:\\Users\\dschw\\Documents\\GitHub\\Thesis\\Output',
                     spacy_model='de_core_news_lg')

nccr_df.generate_bt_corpus()

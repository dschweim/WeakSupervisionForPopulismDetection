import pandas as pd
import time
import os
import glob
from lxml import etree
from util import retrieve_year, standardize_party_naming
pd.options.mode.chained_assignment = None


class BT_Dataset:
    def __init__(
            self,
            data_path: str,
            output_path: str
    ):
        """
        Class to create and pre-process the Bundestag opendata
        :param data_path: path to data input
        :type data_path: str
        :param data_path: path to data output
        :type data_path: str
        """

        self.data_path = data_path
        self.output_path = output_path

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

            # Get header data of file
            speech_date = root.find('.//kopfdaten/veranstaltungsdaten/datum').attrib.get('date')

            speech_location = root.find('.//kopfdaten/veranstaltungsdaten/ort')
            if speech_location is not None:
                speech_location = speech_location.text

            speech_elec_period = root.find('.//kopfdaten/plenarprotokoll-nummer/wahlperiode')
            if speech_elec_period is not None:
                speech_elec_period =  speech_elec_period.text

            speech_session_nr = root.find('.//kopfdaten/plenarprotokoll-nummer/sitzungsnr')
            if speech_session_nr is not None:
                speech_session_nr = speech_session_nr.text

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
                    spr_name_affix = None
                    spr_lastname = None
                    spr_location_affix = None
                    spr_name = None
                    spr_party = None
                    spr_role_full = None
                    spr_role_short = None
                    spr_state = None

                    # If speaker is described by p node, extract values
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
                                        elif ggchild.tag == 'namenszusatz':
                                            spr_name_affix = ggchild.text
                                        elif ggchild.tag == 'nachname':
                                            spr_lastname = ggchild.text
                                        elif ggchild.tag == 'ortszusatz':
                                            spr_location_affix = ggchild.text
                                        elif ggchild.tag == 'fraktion':
                                            spr_party = ggchild.text
                                        elif ggchild.tag == 'rolle':
                                            for gggchild in ggchild.getchildren():
                                                if gggchild.tag == 'rolle_lang':
                                                    spr_role_full = gggchild.text
                                                elif gggchild.tag == 'rolle_kurz':
                                                    spr_role_short = gggchild.text
                                        elif ggchild.tag == 'bdland':
                                            spr_state = ggchild.text

                        # Generate joined name attribute
                        spr_name = ' '.join(filter(None, (spr_title, spr_firstname, spr_name_affix, spr_lastname)))

                    # If speaker is described by name node, extract text
                    elif spr.tag == 'name':
                        spr_name = spr.text

                    # Get count for xpath
                    n = str(index + 1)

                    # Extract text nodes that belong to current subspeech
                    spr_text_j1 = subspeech.xpath('parent::*/p[@klasse=\'J_1\'][' + n + ']')[0].text
                    spr_text_p_list = subspeech.xpath(
                        'parent::*/p[@klasse=\'J_1\'][' + n + ']/following-sibling::p[not (@klasse=\'J_1\')'
                                                              ' and not (@klasse=\'T\') and count(preceding-sibling::p'
                                                              '[@klasse=\'J_1\'])=' + n + ']')

                    # Replace list items with their text content
                    for i, node in enumerate(spr_text_p_list):
                        spr_text_p_list[i] = node.text

                    # Remove NaN from list
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

                    # Generate df for current subspeech
                    df_speech = pd.DataFrame({'text_id': [speech_id],
                                              'text_subid': [subspeech_id],
                                              'text_date': [speech_date],
                                              'text_election_period': [speech_elec_period],
                                              'text_session_nr': [speech_session_nr],
                                              'text_location': [speech_location],
                                              'text_source': [file],
                                              'spr_id': [spr_id],
                                              'spr_name': [spr_name],
                                              'spr_party': [spr_party],
                                              'spr_location_affix': [spr_location_affix],
                                              'spr_role_full': [spr_role_full],
                                              'spr_role_short': [spr_role_short],
                                              'spr_state': [spr_state],
                                              'spr_text': [spr_text]
                                              })

                    # Append result to global corpus
                    df = df.append(df_speech)

        # Save concatenated texts
        df.to_csv(f'{self.output_path}\\BT_corpus.csv', index=True)

        end = time.time()
        print(end - start)
        print('finished BT corpus generation')

    def preprocess_bt_corpus(self, df):

        # Drop rows without speaker_id
        df_prep = df.dropna(subset=["spr_id"])

        # Extract year from date
        df_prep['text_date'] = df_prep['text_date'].astype(str)
        df_prep['year'] = df_prep['text_date'].apply(lambda x: retrieve_year(x))
        df_prep['year'] = df_prep['year'].astype(int)

        # Replace false party name
        df_prep['spr_party'] = df_prep['spr_party'].astype(str)
        df_prep['spr_party'].replace({"Bremen": "SPD",
                                      "Fraktionslos": None,
                                      "fraktionslos": None,
                                      "nan": None}, inplace=True)

        # Standardize party naming
        df_prep['spr_party'] = df_prep['spr_party'].apply(lambda x: standardize_party_naming(x))


        # Save to disk
        df_prep.to_csv(f'{self.output_path}\\BT_corpus_prep.csv', index=True)

        return df_prep

# This main file contains function calls for all other python files
# In order to avoid long runtime set parameters to false after the data output has been initially created
import pandas as pd
import sys
from argparse import ArgumentParser
from NCCR_Corpus_Generator import NCCR_Dataset
from Dict_Generator import Dict_Generator
from Labeling_Framework import Labeler
from BT_Corpus_Generator import BT_Dataset
from util import generate_train_test_split, generate_train_dev_test_split
pd.options.mode.chained_assignment = None

sys.path.append("..")


def main(path_to_project_folder: str,
         spacy_model: str,
         generate_nccr_data: bool,
         preprocess_nccr_data: bool,
         generate_tfidf_dicts: bool,
         generate_chisquare_dict: bool,
         generate_labeling: bool,
         generate_bt_data: bool):
    """
    main function to run project and initialize classes
    :param path_to_project_folder: Trainset
    :type path_to_project_folder: str
    :param spacy_model: used trained Spacy pipeline
    :type: str
    :param generate_nccr_data: Indicator whether to generate data corpus in current run
    :type generate_nccr_data: bool
    :param preprocess_nccr_data: Indicator whether to preprocess data corpus in current run
    :type preprocess_nccr_data: bool
    :param generate_tfidf_dicts: Indicator whether to generate tf-idf dictionaries in current run
    :type generate_tfidf_dicts:  bool
    :param generate_chisquare_dict: Indicator whether to generate chi-square dictionary in current run
    :type generate_chisquare_dict:  bool
    :param generate_labeling: Indicator whether to generate labels from Snorkel in current run
    :type generate_labeling:  bool
     :param generate_bt_data: Indicator whether to generate bundestag data corpus in current run
    :type generate_bt_data: bool
    :return:
    :rtype:
    """

    # Initialize
    nccr_df = NCCR_Dataset(data_path=f'{path_to_project_folder}\\Data',
                           output_path=f'{path_to_project_folder}\\Output',
                           spacy_model=spacy_model)

    nccr_dicts = Dict_Generator(data_path=f'{path_to_project_folder}\\Data',
                                output_path=f'{path_to_project_folder}\\Output\\Dicts',
                                spacy_model=spacy_model)

    if generate_nccr_data:
        # Generate Labelled NCCR
        nccr_data_de_wording_av = nccr_df.generate_labelled_nccr_corpus()

    else:
        # Import corpora
        nccr_data_de_wording_av = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_available.csv')

    if preprocess_nccr_data:
        # Preprocess corpus and generate train-test-split
        nccr_df_prep = nccr_df.preprocess_corpus(nccr_data_de_wording_av)
        #train, test = generate_train_test_split(nccr_df_prep)

        # Generate Train, Dev, Test Split
        train, dev, test = generate_train_dev_test_split(nccr_df_prep)

    else:
        # Import preprocessed corpus and generate train-test-split
        nccr_df_prep = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_available_prep.csv')
        train, dev, test = generate_train_dev_test_split(nccr_df_prep)
        #train, test = generate_train_test_split(nccr_df_prep)

    if generate_tfidf_dicts:
        # Generate Dictionaries based on tfidf
        tfidf_dict = nccr_dicts.generate_tfidf_dict(train, n_words=30)
        tfidf_dict_country = nccr_dicts.generate_tfidf_dict_per_country(train, n_words=30)
        tfidf_dict_global = nccr_dicts.generate_global_tfidf_dict(train, n_words=30)

    else:
        # Import dictionaries
        tfidf_dict = pd.read_csv(f'{path_to_project_folder}\\Output\\Dicts\\tfidf_dict.csv')
        # Convert terms to string
        tfidf_dict.term = tfidf_dict.term.astype(str)

        # Import dictionaries
        tfidf_dict_country_au = pd.read_csv(f'{path_to_project_folder}\\Output\\Dicts\\tfidf_dict_per_country_au.csv')
        tfidf_dict_country_ch = pd.read_csv(f'{path_to_project_folder}\\Output\\Dicts\\tfidf_dict_per_country_ch.csv')
        tfidf_dict_country_de = pd.read_csv(f'{path_to_project_folder}\\Output\\Dicts\\tfidf_dict_per_country_de.csv')
        # Convert terms to string
        tfidf_dict_country_au.term = tfidf_dict_country_au.term.astype(str)
        tfidf_dict_country_ch.term = tfidf_dict_country_ch.term.astype(str)
        tfidf_dict_country_de.term = tfidf_dict_country_de.term.astype(str)

        # Combine in country-dict
        tfidf_dict_country = {}
        values = {'au': tfidf_dict_country_au,
                  'cd': tfidf_dict_country_ch,
                  'de': tfidf_dict_country_de}
        tfidf_dict_country.update(values)

        # Import dictionaries
        tfidf_dict_global = pd.read_csv(f'{path_to_project_folder}\\Output\\Dicts\\tfidf_dict_global.csv')
        # Convert terms to string
        tfidf_dict_global.term = tfidf_dict_global.term.astype(str)

    if generate_chisquare_dict:
        # Generate Dictionary based on chi-square test
        chisquare_dict_global = nccr_dicts.generate_global_chisquare_dict(train, confidence=0.99)
        chisquare_dict_country = nccr_dicts.generate_chisquare_dict_per_country(train, confidence=0.99)

        chisquare_dicts_pop, chisquare_dicts_nonpop = \
            nccr_dicts.generate_chisquare_dep_dicts(train, preprocessed=preprocess_nccr_data, confidence=0.75)

    else:
        # Import dictionary
        chisquare_dict_global = pd.read_csv(f'{path_to_project_folder}\\Output\\Dicts\\chisquare_dict_global.csv')
        # Convert terms to string
        chisquare_dict_global.term = chisquare_dict_global.term.astype(str)

        chisquare_dict_country_au = \
            pd.read_csv(f'{path_to_project_folder}\\Output\\Dicts\\chisquare_dict_per_country_au.csv')
        chisquare_dict_country_ch = \
            pd.read_csv(f'{path_to_project_folder}\\Output\\Dicts\\chisquare_dict_per_country_ch.csv')
        chisquare_dict_country_de = \
            pd.read_csv(f'{path_to_project_folder}\\Output\\Dicts\\chisquare_dict_per_country_de.csv')

        # Convert terms to string
        chisquare_dict_country_au.term = chisquare_dict_country_au.term.astype(str)
        chisquare_dict_country_ch.term = chisquare_dict_country_ch.term.astype(str)
        chisquare_dict_country_de.term = chisquare_dict_country_de.term.astype(str)

        # Combine in country-dict
        chisquare_dict_country = {}
        values = {'au': chisquare_dict_country_au,
                  'cd': chisquare_dict_country_ch,
                  'de': chisquare_dict_country_de}
        chisquare_dict_country.update(values)

    if generate_labeling:
        # Generate overall dictionary as labeling function input
        lf_dict = {'tfidf_keywords': tfidf_dict.term.to_list(),
                   'tfidf_keywords_at': tfidf_dict_country['au'].term.to_list(),
                   'tfidf_keywords_ch': tfidf_dict_country['cd'].term.to_list(),
                   'tfidf_keywords_de': tfidf_dict_country['de'].term.to_list(),
                   'tfidf_keywords_global': tfidf_dict_global.term.to_list(),
                   'chi2_keywords_global': chisquare_dict_global.term.tolist(),
                   'chi2_keywords_at': chisquare_dict_country['au'].term.to_list(),
                   'chi2_keywords_ch': chisquare_dict_country['cd'].term.to_list(),
                   'chi2_keywords_de': chisquare_dict_country['de'].term.to_list(),
                   'chi2_dicts_pop': chisquare_dicts_pop,
                   'chi2_dicts_nonpop': chisquare_dicts_nonpop
                   }

        # Filter on relevant columns
        train_sub = train[['ID', 'wording_segments', 'party', 'Sample_Country', 'year', 'POPULIST',
                           'POPULIST_PeopleCent', 'POPULIST_AntiElite', 'POPULIST_Sovereign']]
        test_sub = test[['ID', 'wording_segments', 'party', 'Sample_Country', 'year', 'POPULIST',
                         'POPULIST_PeopleCent', 'POPULIST_AntiElite', 'POPULIST_Sovereign']]
        dev_sub = dev[['ID', 'wording_segments', 'party', 'Sample_Country', 'year', 'POPULIST',
                       'POPULIST_PeopleCent', 'POPULIST_AntiElite', 'POPULIST_Sovereign']]
        train_sub.rename({'wording_segments': 'text'}, axis=1, inplace=True)
        test_sub.rename({'wording_segments': 'text'}, axis=1, inplace=True)
        dev_sub.rename({'wording_segments': 'text'}, axis=1, inplace=True)

        print('TRAIN EXAMPLES: ' + str(len(train)))
        print('DEV EXAMPLES: ' + str(len(dev)))
        print('TEST EXAMPLES: ' + str(len(test)))

        # Initialize Labeler
        nccr_labeler = Labeler(train_data=train_sub,
                               test_data=test_sub,
                               dev_data = dev_sub,
                               lf_input_dict=lf_dict,
                               data_path=f'{path_to_project_folder}\\Data',
                               output_path=f'{path_to_project_folder}\\Output')

        # Run Snorkel Labeling
        nccr_labeler.run_labeling()

    if generate_bt_data:
        bt_df = BT_Dataset(data_path=f'{path_to_project_folder}\\Data',
                           output_path=f'{path_to_project_folder}\\Output')

        bt_df.generate_bt_corpus()

    else:
        bt_df = pd.read_csv(f'{path_to_project_folder}\\Output\\BT_corpus.csv')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input

    main(path_to_project_folder=input_path,
         spacy_model='de_core_news_lg',  #de_dep_news_trf
         generate_nccr_data=False,
         preprocess_nccr_data=False,  # runs for approx 15 min
         generate_tfidf_dicts=False,
         generate_chisquare_dict=False,
         generate_labeling=False,
         generate_bt_data=True)

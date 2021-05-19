# This main file contains function calls for all other python files
# In order to avoid long runtime set parameters to false after the data output has been initially created
import pandas as pd
import sys
from argparse import ArgumentParser
from NCCR_Corpus import PCCR_Dataset
from Labeling_Framework import Labeler
from util import generate_train_test_split, generate_train_dev_test_split

sys.path.append("..")


def main(path_to_project_folder: str,
         generate_data: bool, generate_train_test: bool,
         generate_tfidf_dicts: bool, generate_labeling: bool):
    """
    main function to run project and initialize classes
    :param path_to_project_folder: Trainset
    :type path_to_project_folder: str
    :param generate_data: Indicator whether to generate data corpus in current run
    :type generate_data: bool
    :param generate_train_test:  Indicator whether to generate train test split in current run
    :type generate_train_test:  bool
    :param generate_tfidf_dicts: Indicator whether to generate tf-idf dictionaries in current run
    :type generate_tfidf_dicts:  bool
    :param generate_labeling: Indicator whether to generate labels from Snorkel in current run
    :type generate_labeling:  bool
    :return:
    :rtype:
    """

    # Initialize
    nccr_df = PCCR_Dataset(data_path=f'{path_to_project_folder}\\Data',
                           output_path=f'{path_to_project_folder}\\Output')

    if generate_data:
        # Generate Labelled NCCR
        nccr_data_de_wording_all, nccr_data_de_wording_av = nccr_df.generate_labelled_nccr_corpus()

    else:
        # Import corpora
        nccr_data_de_wording_all = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_all.csv'
        )
        nccr_data_de_wording_av = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_available.csv'
        )

    if generate_train_test:
        # Generate Train, Test Split
        train, test = generate_train_test_split(nccr_data_de_wording_av)
        # Pre-process data
        train_prep = nccr_df.preprocess_corpus(train, is_train=True)
        test_prep = nccr_df.preprocess_corpus(test, is_train=False)

        # Generate Train, Dev, Test Split
        # train, dev, test = generate_train_dev_test_split(nccr_data_de_wording_av)

    else:
        # Import preprocessed data
        train_prep = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_available_TRAIN.csv'
        )
        test_prep = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_available_TEST.csv'
        )

    if generate_tfidf_dicts:
        # Generate Dictionaries based on tfidf
        tfidf_dict = nccr_df.generate_tfidf_dict(train_prep, tfidf_threshold=30)
        tfidf_dict_country = nccr_df.generate_tfidf_dict_per_country(train_prep, tfidf_threshold=30)
        tfidf_dict_global = nccr_df.generate_global_tfidf_dict(train_prep, tfidf_threshold=30)

    else:
        # Import dictionaries
        tfidf_dict = pd.read_csv(f'{path_to_project_folder}\\Output\\tfidf_dict.csv')

        tfidf_dict_country_au = pd.read_csv(f'{path_to_project_folder}\\Output\\tfidf_dict_per_country_au.csv')
        tfidf_dict_country_ch = pd.read_csv(f'{path_to_project_folder}\\Output\\tfidf_dict_per_country_ch.csv')
        tfidf_dict_country_de = pd.read_csv(f'{path_to_project_folder}\\Output\\tfidf_dict_per_country_de.csv')

        tfidf_dict_country = {}
        values = {'au': tfidf_dict_country_au,
                  'cd': tfidf_dict_country_ch,
                  'de': tfidf_dict_country_de}
        tfidf_dict_country.update(values)

        tfidf_dict_global = pd.read_csv(f'{path_to_project_folder}\\Output\\tfidf_dict_global.csv')

    if generate_labeling:
        # Generate overall dictionary as labeling function input
        lf_dict = {'tfidf_keywords': tfidf_dict.term.to_list(),
                   'tfidf_keywords_at': tfidf_dict_country['au'].term.to_list(),
                   'tfidf_keywords_ch': tfidf_dict_country['cd'].term.to_list(),
                   'tfidf_keywords_de': tfidf_dict_country['de'].term.to_list(),
                   'tfidf_keywords_global': tfidf_dict_global.term.to_list()}

        # Filter on relevant columns
        train_prep_sub = train_prep[['ID', 'text_prep', 'party', 'Sample_Country', 'year', 'POPULIST']]
        test_prep_sub = test_prep[['ID', 'text_prep', 'party', 'Sample_Country', 'year', 'POPULIST']]
        train_prep_sub.rename({'text_prep': 'text'}, axis=1, inplace=True)
        test_prep_sub.rename({'text_prep': 'text'}, axis=1, inplace=True)

        #train_prep_sub.drop_duplicates(subset='ID', keep=False, inplace=True)
        #test_prep_sub.drop_duplicates(subset='ID', keep=False, inplace=True)

        # Initialize Labeler
        nccr_labeler = Labeler(train_data=train_prep_sub,
                               test_data=test_prep_sub,
                               lf_input_dict=lf_dict,
                               data_path=f'{path_to_project_folder}\\Data',
                               output_path=f'{path_to_project_folder}\\Output')

        # Run Snorkel Labeling
        nccr_labeler.run_labeling()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input

    main(path_to_project_folder=input_path,
         generate_data=True,
         generate_train_test=False,
         generate_tfidf_dicts=False,
         generate_labeling=True)

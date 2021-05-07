# This main file contains function calls for all other python files
# In order to avoid long runtime set parameters to false after the data output has been initially created
import pandas as pd
import sys
from argparse import ArgumentParser
from NCCR_Corpus import PCCR_Dataset
from Labeling_Framework import snorkel_labeling
from util import generate_train_test_split, generate_train_dev_test_split

sys.path.append("..")


def main(input_path, generate_data, run_labeling, generate_train_test, generate_tfidf_dicts):
    # Set project path globally
    path_to_project_folder = input_path

    # Initialize
    df_nccr = PCCR_Dataset(data_path=f'{path_to_project_folder}\\Data',
                           output_path=f'{path_to_project_folder}\\Output')

    # Either generate data or read data from disk
    if generate_data:
        # Generate Labelled NCCR
        nccr_data_de_wording_all, nccr_data_de_wording_av = df_nccr.generate_labelled_nccr_corpus()

    else:
        # Import corpora
        nccr_data_de_wording_all = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_all.csv'
        )
        nccr_data_de_wording_av = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_available.csv'
        )

    # Run Snorkel framework if set
    if run_labeling:

        # Either generate train test split or read from disk
        if generate_train_test:
            # Generate Train, Test Split
            train, test = generate_train_test_split(nccr_data_de_wording_av)
            # Pre-process data
            train_prep = df_nccr.preprocess_corpus(train, is_train=True)
            test_prep = df_nccr.preprocess_corpus(test, is_train=False)

            # Generate Train, Dev, Test Split
            #train, dev, test = generate_train_dev_test_split(nccr_data_de_wording_av)

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
            tfidf_dict = df_nccr.generate_tfidf_dict(train_prep, tfidf_threshold=0.01)
            tfidf_dict_country = df_nccr.generate_tfidf_dict_per_country(train_prep, tfidf_threshold=0.01)
            tfidf_dict_global = df_nccr.generate_global_tfidf_dict(train_prep, tfidf_threshold=0.1)

        else:
            # Import dictionaries
            tfidf_dict = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict.csv'
            )

            tfidf_dict_country_au = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict_per_country_au.csv'
            )
            tfidf_dict_country_ch = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict_per_country_ch.csv'
            )
            tfidf_dict_country_de = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict_per_country_de.csv'
            )

            tfidf_dict_country = {}
            values = {'au': tfidf_dict_country_au,
                      'cd': tfidf_dict_country_ch,
                      'de': tfidf_dict_country_de}
            tfidf_dict_country.update(values)

            tfidf_dict_global = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict_global.csv'
            )


        # Generate overall dictionary as labeling function input
        lf_dict = {'tfidf_keywords': tfidf_dict.term.to_list(),
                   'tfidf_keywords_at': tfidf_dict_country['au'].term.to_list(),
                   'tfidf_keywords_ch': tfidf_dict_country['cd'].term.to_list(),
                   'tfidf_keywords_de': tfidf_dict_country['de'].term.to_list(),
                   'tfidf_keywords_global': tfidf_dict_global.term.to_list()}

        # Filter on relevant columns
        train_prep_sub = train_prep[['text_prep', 'party', 'Sample_Country', 'year', 'POPULIST']]
        test_prep_sub = test_prep[['text_prep', 'party', 'Sample_Country', 'year', 'POPULIST']]
        train_prep_sub.rename({'text_prep': 'text'}, axis=1, inplace=True)
        test_prep_sub.rename({'text_prep': 'text'}, axis=1, inplace=True)

        # Run Snorkel framework
        snorkel_labeling(train_data=train_prep_sub,
                         test_data=test_prep_sub,
                         lf_input_dict=lf_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input

    main(input_path=input_path,
         generate_data=False,
         run_labeling=True,
         generate_train_test=False,
         generate_tfidf_dicts=False)

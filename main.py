import pandas as pd
from NCCR_Pred import PCCR_Dataset
from Labeling import snorkel_labeling
from util import generate_train_test_split


def main(generate_data, run_labeling, generate_train_test, generate_tfidf_dicts):
    # Initialize
    df_nccr = PCCR_Dataset(data_path="C:/Users/dschw/Documents/GitHub/Thesis/Data",
                           output_path="C:/Users/dschw/Documents/GitHub/Thesis/Output")

    # Either generate data or read data from disk
    if generate_data:
        # Generate Labelled NCCR
        nccr_data_de_wording_av, nccr_data_de_wording_all = df_nccr.generate_labelled_nccr_corpus()

    else:
        # Import corpora
        nccr_data_de_wording_all = pd.read_csv(
            "C:/Users/dschw/Documents/GitHub/Thesis/Output/NCCR_combined_corpus_DE_wording_all.csv"
        )
        nccr_data_de_wording_av = pd.read_csv (
            "C:/Users/dschw/Documents/GitHub/Thesis/Output/NCCR_combined_corpus_DE_wording_available.csv"
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

        else:
            # Import preprocessed data
            train_prep = pd.read_csv(
                "C:/Users/dschw/Documents/GitHub/Thesis/Output/NCCR_combined_corpus_DE_wording_available_TRAIN.csv"
            )
            test_prep = pd.read_csv(
                "C:/Users/dschw/Documents/GitHub/Thesis/Output/NCCR_combined_corpus_DE_wording_available_TEST.csv"
            )

        if generate_tfidf_dicts:
            # Generate Dictionaries based on tfidf
            tfidf_dict = df_nccr.generate_tfidf_dict(train_prep, tfidf_threshold=0.01)
            tfidf_dict_country = df_nccr.generate_tfidf_dict_per_country(train_prep, tfidf_threshold=0.01)
            tfidf_dict_global = df_nccr.generate_global_tfidf_dict(train_prep, tfidf_threshold=0.1)

        else:
            # Import dictionaries
            tfidf_dict = pd.read_csv(
                "C:/Users/dschw/Documents/GitHub/Thesis/Output/tfidf_dict.csv"
            )


            tfidf_dict_country_au = pd.read_csv(
                "C:/Users/dschw/Documents/GitHub/Thesis/Output/tfidf_dict_per_country_au.csv"
            )
            tfidf_dict_country_ch = pd.read_csv(
                "C:/Users/dschw/Documents/GitHub/Thesis/Output/tfidf_dict_per_country_ch.csv"
            )
            tfidf_dict_country_de = pd.read_csv(
                "C:/Users/dschw/Documents/GitHub/Thesis/Output/tfidf_dict_per_country_de.csv"
            )

            tfidf_dict_country = {}
            values = {'au': tfidf_dict_country_au,
                      'cd': tfidf_dict_country_ch,
                      'de': tfidf_dict_country_de}
            tfidf_dict_country.update(values)


            tfidf_dict_global = pd.read_csv(
                "C:/Users/dschw/Documents/GitHub/Thesis/Output/tfidf_dict_global.csv"
            )

        # Generate overall dictionary as labeling function input
        lf_dict = {'tfidf_keywords': tfidf_dict.term.to_list(),
                   'tfidf_keywords_at': tfidf_dict_country['au'].term.to_list(),
                   'tfidf_keywords_ch': tfidf_dict_country['cd'].term.to_list(),
                   'tfidf_keywords_de': tfidf_dict_country['de'].term.to_list(),
                   'tfidf_keywords_global': tfidf_dict_global.term.to_list()}

        # Filter on relevant columns
        train_prep_sub = train_prep[['text_prep', 'POPULIST']]
        test_prep_sub = test_prep[['text_prep', 'POPULIST']]
        train_prep_sub.rename({'text_prep': 'text'}, axis=1, inplace=True)
        test_prep_sub.rename({'text_prep': 'text'}, axis=1, inplace=True)

        # Run Snorkel framework
        snorkel_labeling(train_data=train_prep_sub,
                         test_data=test_prep_sub,
                         lf_input=lf_dict)


main(generate_data=False, run_labeling=True,
     generate_train_test=False, generate_tfidf_dicts=False)

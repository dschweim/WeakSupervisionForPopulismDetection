import time

import pandas as pd

from NCCR_Pred import PCCR_Dataset
from Labeling import snorkel_labeling
from util import generate_train_test_split


def main(generate_data, preprocess_data, run_labeling):
    # Initialize
    df_nccr = PCCR_Dataset(data_path="C:/Users/dschw/Documents/GitHub/Thesis/Data",
                           output_path="C:/Users/dschw/Documents/GitHub/Thesis/Output")

    # Either generate data or read data from disk
    if generate_data:
        # Generate Labelled NCCR
        nccr_data_de = df_nccr.generate_labelled_nccr_corpus()

    else:
        # Import labelled nccr
        nccr_data_de = pd.read_csv("C:/Users/dschw/Documents/GitHub/Thesis/Output/labelled_nccr_corpus_DE.csv")

    # Generate Train, Test Split
    train, test = generate_train_test_split(nccr_data_de)

    if preprocess_data:
        # Pre-process data
        train_prep = df_nccr.preprocess_corpus(train, is_train=True)
        test_prep = df_nccr.preprocess_corpus(test, is_train=False)

    else:
        # Import preprocessed data
        train_prep = pd.read_csv("C:/Users/dschw/Documents/GitHub/Thesis/Output/labelled_nccr_corpus_DE_TRAIN.csv")
        test_prep = pd.read_csv("C:/Users/dschw/Documents/GitHub/Thesis/Output/labelled_nccr_corpus_DE_TEST.csv")

    # todo: Generate Dictionary based on tfidf
    tfidf_dict = df_nccr.generate_tfidf_dict(train)

    # Run Snorkel framework if set
    if run_labeling:

        # Run Snorkel framework
        snorkel_labeling(train_prep[['text_prep', 'POPULIST']],
                         test_prep[['text_prep', 'POPULIST']])


main(generate_data=False, preprocess_data=False, run_labeling=True)

import pandas as pd

from NCCR_Pred import PCCR_Dataset
from Labeling import snorkel_labeling


def main(generate_data, run_labeling):
    # Initialize
    df_nccr = PCCR_Dataset(data_path="C:/Users/dschw/Documents/GitHub/Thesis/Data",
                           output_path="C:/Users/dschw/Documents/GitHub/Thesis/Output")

    # Either generate data or read data from disk
    if generate_data:
        # Generate Labelled NCCR
        data_de = df_nccr.generate_labelled_nccr_corpus()

    else:
        # Import labelled nccr
        data_de = pd.read_csv("C:/Users/dschw/Documents/GitHub/Thesis/Output/labelled_nccr_corpus_DE.csv")

    # Generate Train, Test Split
    train, test = df_nccr.generate_train_test_split(data_de)

    # Run Snorkel framework if set
    if run_labeling:
        # Filter on relevant columns for labeling
        train_prep = train[['text', 'POPULIST']]
        test_prep = test[['text', 'POPULIST']]

        # Run Snorkel framework
        snorkel_labeling(train_prep, test_prep)

    # Generate TFIDF
    train_prep = df_nccr.preprocess_corpus(train)
    #test_prep = df_nccr.preprocess_corpus(test)

main(generate_data=False, run_labeling=False)

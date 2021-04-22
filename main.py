import pandas as pd

from argparse import ArgumentParser

from NCCR_Pred import PCCR_Dataset
from Labeling import snorkel_labeling


def main(generate_data, run_labeling):
    # Initialize
    df_nccr = PCCR_Dataset(data_path="C:/Users/dschw/Documents/GitHub/Thesis/Data",
                           output_path="C:/Users/dschw/Documents/GitHub/Thesis/Output")

    if generate_data:
        # Generate Labelled NCCR
        data_de = df_nccr.generate_labelled_NCCR_corpus()

    else:
        # Import labelled nccr
        data_de = pd.read_csv("C:/Users/dschw/Documents/GitHub/Thesis/Output/labelled_nccr_corpus_DE.csv")

    if run_labeling:
        # Filter on relevant columns for labeling
        data_de_labeling_subcorpus = data_de[['text', 'POPULIST']]
        train, test = df_nccr.generate_train_test_split(data_de_labeling_subcorpus)
        snorkel_labeling(train, test)

main(generate_data=True, run_labeling=True)

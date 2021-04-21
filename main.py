import pandas as pd

from argparse import ArgumentParser

from NCCR_Pred import PCCR_Dataset


def main(generate_data):
    # Initialize
    df_nccr = PCCR_Dataset(data_path="C:/Users/dschw/Documents/GitHub/Thesis/Data",
                           output_path="C:/Users/dschw/Documents/GitHub/Thesis/Output")

    if generate_data:
        # Generate Labelled NCCR
        data_de = df_nccr.generate_labelled_NCCR_corpus()

    else:
        # Import labelled nccr
        data_de = pd.read_csv("C:/Users/dschw/Documents/GitHub/Thesis/Output/labelled_nccr_corpus_DE.csv")

    # todo: Temp. generate sub-corpus (one-example per cat)
    data_de_sub = data_de.loc[(data_de['POPULIST_PeopleCent'] == 1) |
                              (data_de['POPULIST_AntiElite'] == 1) |
                              (data_de['POPULIST_Sovereign'] == 1)]

    #data_prep = df_nccr.preprocess_corpus(data_de_sub)
    # print(data_train_prep)


main(generate_data=True)

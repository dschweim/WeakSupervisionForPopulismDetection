from typing import List
import random
import pandas as pd
import numpy as np
import json
import pathlib
import torch
from argparse import ArgumentParser

from util import generate_train_test_split

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.model_selection import train_test_split

def run_transformer(train_data, test_data):

        # Set model global to use it inside model_init function
        global model

        model = "bert-base-german-cased"
        print(model)

        # Read the data
        train_data =train_data
        test_data = test_data

        # Tokenize the text features
        # Instantiate Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)

        # Encode the text features for training (test data is encoded later)
        train_encodings = tokenizer(train_data.wording_segments.tolist(), truncation=True, padding=True)

        # Create Trainset
        #train_set = CustomDataset(train_encodings, train_data.label.tolist())

        # Load Transformer Model
        # Set model global to use it inside model_init function
        global model_config
        model_config = AutoConfig.from_pretrained(model, num_labels=train_data['POPULIST'].nunique())
        transformer_model = AutoModelForSequenceClassification.from_pretrained(model, config=model_config)

        # Create Trainer Object
        # Use GPU, if available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        run_parameter_search = True




 # Import preprocessed corpus and generate train-test-split
nccr_df_prep = pd.read_csv('C:/Users/dschw/Documents/GitHub/Thesis/Output/NCCR_combined_corpus_DE_wording_available_prep.csv')

train, test = generate_train_test_split(nccr_df_prep)
run_transformer(train,test)
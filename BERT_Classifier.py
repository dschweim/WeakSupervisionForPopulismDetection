import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from util import generate_train_test_split


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def run_transformer(train_data, test_data, model_name):
    # Load the data
    train_data = train_data

    # Instantiate Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode the text features for training (test data is encoded later)
    X = train_data.content.tolist()
    y = train_data.label.tolist()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Tokenize
    X_train_tokenized = tokenizer(X_train, padding='max_length', truncation=True)
    X_val_tokenized = tokenizer(X_val, padding='max_length', truncation=True)

    # Create torch dataset
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # Instantiate model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Define Trainer
    training_args = TrainingArguments("test_trainer")
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset
    )

    # Train pre-trained model
    trainer.train()

    # ----- 3. Predict -----#
    # Load test data
    test_data = test_data
    X_test = test_data.content.tolist()
    X_test_tokenized = tokenizer(X_test, padding='max_length', truncation=True)

    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)

    model_path = "output/checkpoint-50000"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)

    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

# Import preprocessed corpus and generate train-test-split
nccr_df_prep = pd.read_csv(
    'C:/Users/dschw/Documents/GitHub/Thesis/Output/NCCR_combined_corpus_DE_wording_available_prep.csv')

train, test = generate_train_test_split(nccr_df_prep)
train.rename({'wording_segments': 'content'}, axis=1, inplace=True)
test.rename({'wording_segments': 'content'}, axis=1, inplace=True)

train.rename({'POPULIST': 'label'}, axis=1, inplace=True)
test.rename({'POPULIST': 'label'}, axis=1, inplace=True)

train['label'] = train['label'].astype(int)
test['label'] = test['label'].astype(int)

train= train.head(10)
test = test.head(3)

run_transformer(train, test, model_name='bert-base-german-cased')

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback


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


def run_transformer(X, y, X_test, model_name):

    # Instantiate Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode the text features for training (test data is encoded later)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Tokenize
    X_train_tokenized = tokenizer(X_train, padding='max_length', truncation=True)
    X_val_tokenized = tokenizer(X_val, padding='max_length', truncation=True)

    # Create torch dataset
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # Instantiate model
    transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Create Trainer Object
    # Use GPU, if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define Trainer
    training_args = TrainingArguments("test_trainer")
    trainer = Trainer(
        model=transformer_model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset
    )

    # Train pre-trained model
    transformer_model.to(device)
    trainer.train()

    # ----- 3. Predict -----#
    # Load test data
    X_test_tokenized = tokenizer(X_test, padding='max_length', truncation=True)

    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)

    # Make prediction
    raw_pred = trainer.predict(test_dataset)

    # Preprocess raw predictions
    y_pred = raw_pred[0].argmax(-1)

    # Get hyperparams
    hyperparameters = {
        "batch_size": trainer.args.per_device_train_batch_size,
        "learning_rate": trainer.args.learning_rate,
        "num_epochs": trainer.args.num_train_epochs}

    return y_pred, hyperparameters
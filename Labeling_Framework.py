import pandas as pd
from snorkel.labeling import PandasLFApplier, LFAnalysis, filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from snorkel.utils import probs_to_preds
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from Labeling_Functions import get_lfs


def snorkel_labeling(train_data: pd.DataFrame, test_data: pd.DataFrame, lf_input_dict: dict):
    """
    Calculate tf-idf scores of docs and return top n words that
    :param train_data: Trainset
    :type train_data:  DataFrame
    :param test_data: Testset
    :type test_data:  DataFrame
    :param lf_input_dict: Dictionary with further input for labeling functions
    :type test_data:  dict
    :return: Returns labelled Dataset
    :rtype:  DataFrame
    """

    ## 1. Define labeling functions
    # Retrieve defined labeling functions
    lfs = get_lfs(lf_input_dict)

    ## 2. Generate label matrix L
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=train_data)

    # Evaluate performance on training set
    print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    ## 3. Generate label model
    # Baseline: Majority Model
    majority_model = MajorityLabelVoter()
    preds_train = majority_model.predict(L=L_train)

    # Advanced: Label Model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    # Extract target label
    Y_test = test_data['POPULIST']
    L_test = applier.apply(df=test_data)

    majority_scores_dict = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random",
                                                metrics=["accuracy", "precision", "recall"])

    print(f"{'Majority Vote Accuracy:':<25} {majority_scores_dict['accuracy']}")
    print(f"{'Majority Vote Precision:':<25} {majority_scores_dict['precision']}")
    print(f"{'Majority Vote Recall:':<25} {majority_scores_dict['recall']}")

    label_model_scores_dict = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random",
                                                metrics=["accuracy", "precision", "recall"])

    print(f"{'Label Model Accuracy:':<25} {label_model_scores_dict['accuracy']}")
    print(f"{'Label Model Precision:':<25} {label_model_scores_dict['precision']}")
    print(f"{'Label Model Recall:':<25} {label_model_scores_dict['recall']}")

    # todo: Classifier
    ## 4. Train classifier
    # Filter out unlabeled data points
    probs_train = label_model.predict_proba(L=L_train)
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=train_data, y=probs_train, L=L_train
    )

    vectorizer = CountVectorizer(ngram_range=(1, 5))
    X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
    X_test = vectorizer.transform(test_data.text.tolist())

    preds_train_filtered = probs_to_preds(probs=probs_train_filtered)
    sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
    sklearn_model.fit(X=X_train, y=preds_train_filtered)

    print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")

import pandas as pd
#from utils import load_unlabeled_spam_dataset
from snorkel.labeling import labeling_function


# 1. Writing Labeling Functions

#df_train = load_unlabeled_spam_dataset()

# Define the label mappings for convenience
ABSTAIN = -1
NOT_POP = 0
POP = 1

# Regular expression for
@labeling_function()
def keyword_match_we(x):



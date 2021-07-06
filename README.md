# Master Thesis: Extraction of a Dataset for Populism Detection using Weak Supervision

This repository contains the data and code to replicate the result of the above mentioned Master Thesis.

## Setup
*--i path_to_file*

## Replication Material


### Data
The folder *data* contains all necessary input sources in the respective subfolder:
- *BT_OpenData*: Bundestag plenary minutes of the 19th legislative period in xml-format, downloaded from [bundestag.de](https://www.bundestag.de/services/opendata)
- *CHES*: Chapel Hill Expert Survey (CHES) files for 2014, 2017, and 2019 in csv-format, downloaded from [chesdata.eu](https://www.chesdata.eu/our-surveys) 
- *NCCR_Content*: 
- *NRC-Emotion-Lexicon*
- *SentiWS_v2.0*

### Notebooks
The folder *Notebooks* contains Jupyter Notebooks that were primarily used for experimentation, but also to generate xx
- `Data_Exploration.ipynb`: Contains the code used to explore the NCCR and BT data and calculate simple dataset statistics
- `Manual_Segment_Extraction.ipynb`: Contains the code used to manually fix non-matched Wording Content to retrieve corresponding Segments from Text.
- `Spay_Exploration.ipynb`: Contains the code used to explore spacy models and word embedding similarities

### Output
The folder *Output* contains the results from each Code file:


### Code
All files can be run from within the 'main.py' file. To this end, corresponding indicators have to be set to true.
The following Python files are included in this repo:

- `BERT_Classifier.py`
- `BT_Corpus_Generator.py`: Code for generating a suitable corpus in csv-format of the Bundestag plenary minutes using the *BT_OpenData* files.
- `Dict_Generator.py`
- `Labeling_Framework.py`
- `Labeling_Functions.py`
- `NCCR_Coprus_Generator.py`: Code for generating a suitable corpus in csv-format of the Bundestag plenary minutes using the *NCCR_Content* files.

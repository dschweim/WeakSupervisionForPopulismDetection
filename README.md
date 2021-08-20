# Master Thesis: Extraction of a Dataset for Populism Detection using Weak Supervision

This folder contains all experimental results, as well as the data and code for replicating the result of the above mentioned Master Thesis.

## Setup
*--i path_to_file*

## Replication Material


### Data
The folder *Data* contains all necessary input sources in the respective subfolder:
- *BT_OpenData*: Bundestag plenary minutes of the 19th legislative period in xml-format, downloaded from [bundestag.de](https://www.bundestag.de/services/opendata).
- *CHES*: Chapel Hill Expert Survey (CHES) files for 2014, 2017, and 2019 in csv-format, downloaded from [chesdata.eu](https://www.chesdata.eu/our-surveys).
- *NCCR_Content*: 
- *NRC-Emotion-Lexicon*: NRC Word-Emotion Association Lexicon in csv-format including instructions, published papers, and further information documents, downloaded from [saifmohammad.com](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm).

### Experiments
The folder *Experiments* Contains full results of all conducted experiments in the respective subfolder:
- *End_Model_Performances*: Results obtained with end models under different setups.
- *Keypattern_Dicts*: Results obtained with labeling framework using keypattern dicts retrieved with different confidence levels.
- *Label_Model_Performances*: Results obtained with label models under different setups.

### Notebooks
The folder *Notebooks* contains Jupyter Notebooks that were primarily used for experimentation, but also to generate xx
- `Data_Exploration.ipynb`: Contains the code used to explore the NCCR and BT data and calculate simple dataset statistics.
- `Figure_Generator.ipynb`
- `Interpretation and Error Analysis End Models.ipynb`
- `Interpretation and Error Analysis Label Models.ipynb`
- `Manual_Segment_Extraction.ipynb`: Contains the code used to manually fix non-matched Wording Content to retrieve corresponding segments from text.
- `Run_Transformer.ipynb`

### Output
The folder *Output* contains the results from each Code file:

### runs
The folder *runs* contains caches of the employed Transformer models.

### Code
All files can be run from within the 'main.py' file. To this end, corresponding indicators have to be set to true.
The following Python files are included in this repo:

- `Transformer_Classifier.py`
- `BT_Corpus_Generator.py`: Code for generating a suitable corpus in csv-format of the Bundestag plenary minutes using the *BT_OpenData* files.
- `Dict_Generator.py`
- `Labeling_Framework.py`
- `Labeling_Functions.py`
- `NCCR_Corpus_Generator.py`: Code for generating a suitable corpus in csv-format of the Bundestag plenary minutes using the *NCCR_Content* files.

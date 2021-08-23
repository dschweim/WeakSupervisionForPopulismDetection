# Master Thesis: Extraction of a Dataset for Populism Detection using Weak Supervision

This folder contains all experimental results, as well as the data and code for replicating the result of the above mentioned Master Thesis.

## Setup
To run the code following code, please make sure to include the path that leads to the folder of this project as input parameter:
*--i path_to_file*

## Replication Material

### Data
The folder *Data* contains all necessary input sources in the respective subfolder:
- *BT_OpenData*: Bundestag plenary minutes of the 19th legislative period in xml-format, downloaded from [bundestag.de](https://www.bundestag.de/services/opendata).
- *CHES*: Chapel Hill Expert Survey (CHES) files for 2014, 2017, and 2019 in csv-format, downloaded from [chesdata.eu](https://www.chesdata.eu/our-surveys).
- *NCCR_Content*: Data from content analysis by the National Centre of Competence in Research (NCCR) in txt- and xml-format, downloaded from [drive.switch.ch](https://drive.switch.ch/index.php/s/ZEaSw5xAkA28nTO).
- *NRC-Emotion-Lexicon*: NRC Word-Emotion Association Lexicon in csv-format including instructions, published papers, and further information documents, downloaded from [saifmohammad.com](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm).

### Experiments
The folder *Experiments* Contains full results of all conducted experiments in the respective subfolder:
- *End_Model_Performances*: Results obtained with end models under different setups.
- *Keypattern_Dicts*: Results obtained with labeling framework using keypattern dicts retrieved with different confidence levels.
- *Label_Model_Performances*: Results obtained with label models under different setups.

### Notebooks
The folder *Notebooks* contains Jupyter Notebooks that were primarily used for experimentation and analysis:
- `Data_Exploration.ipynb`: Contains the code used to explore the NCCR and BT data and calculate simple dataset statistics.
- `Figure_Generator.ipynb`: Used to generate further plots and stats for the report.
- `Interpretation and Error Analysis End Models.ipynb`: Used to generate plots and stats for the individual end models.
- `Interpretation and Error Analysis Label Models.ipynb`: Used to generate plots and stats for the individual label models.
- `Manual_Segment_Extraction.ipynb`: Contains the code used to manually fix non-matched Wording Content to retrieve corresponding segments from text.
- `Run_Transformer.ipynb`: Used to run the Transformer-based models in a GPU-environment with Colab.

### Output
The folder *Output* contains the results from each Code file:

### runs
The folder *runs* contains caches of the employed Transformer models.

### Code
All files can be run from within the 'main.py' file. To this end, corresponding indicators have to be set to true.
The following Python files are included in this repo:

- `BT_Corpus_Generator.py`: Class used for generating and preprocessing a corpus in csv-format of the Bundestag plenary minutes using the *BT_OpenData* files.
- `DEP_Matching.py`: Class used to define the Dependency Matcher along with the key dependency patterns
- `Dict_Generator.py`: Class for generating keyword and keypattern dictionaries
- `Labeling_Framework.py`: Class that incorporates the weak supervision framework based on Snorkel
- `Labeling_Functions.py`: Class used to instantiate the used set of labeling functions based on Snorkel as input for the `Labeling_Framework.py`
- `NCCR_Corpus_Generator.py`: Class used for generating and preprocessing a corpus in csv-format using the *NCCR_Content* files.
- `Transformer_Classifier.py`: Implementation of Transformer-based classifiers to be used as end models.
- `util.py`: contains functions used within oder classes
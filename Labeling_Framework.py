import pandas as pd
import re
import spacy
from snorkel.labeling import PandasLFApplier, LFAnalysis, filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from snorkel.utils import probs_to_preds
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from snorkel.analysis import get_label_buckets
from util import standardize_party_naming, extract_dep_tuples, get_all_svo_tuples, get_svo_tuples_segment
from Labeling_Functions import get_lfs
from DEP_Matcher_Test import test_dep_matcher


class Labeler:
    def __init__(
            self,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            lf_input_dict: dict,
            data_path: str,
            output_path: str,
            spacy_model: str
    ):
        """
        Class to create the labels
        :param train_data: training set
        :type train_data: pd.DataFrame
        :param test_data: test set
        :type test_data: pd.DataFrame
        :param lf_input_dict: dict that contains input for LFs
        :type lf_input_dict: dict
        :param data_path: path to data input
        :type data_path: str
        :param output_path: path to data output
        :type output_path: str
        :param spacy_model: used trained Spacy pipeline
        :type: str
        """

        self.train_data = train_data
        self.test_data = test_data
        self.lf_input_dict = lf_input_dict
        self.data_path = data_path
        self.output_path = output_path
        self.spacy_model = spacy_model

    def run_labeling(self):
        """
        Generate labels on initialized dataframe using Snorkel
        :return:
        :rtype:
        """

        ABSTAIN = -1
        NONPOP = 0
        POP = 1

        # Define train & test data
        train_data = self.train_data
        test_data = self.test_data

        # todo: Pre-process corpora with Dependency Matcher
        #nlp_full = spacy.load(self.spacy_model)
        #train_data['doc'] = list(nlp_full.pipe(train_data['wording_segments'])) #todo: necessary?
        #train_data['doc'].apply(lambda x: test_dep_matcher(x))

        #todo: input whole corpus in test_dep_matcher -> Retrieve corpus with additional columns for each
        # DEP MATCHER indicating whether match exists or not (need seperate matchers to generate separate LFs)


        ## 1. Define labeling functions
        # Generate dict with ches input
        ches_14, ches_17, ches_19 = self.__prepare_labeling_input_ches()
        lf_input_ches = {}
        values = {'ches_14': ches_14,
                  'ches_17': ches_17,
                  'ches_19': ches_19}
        lf_input_ches.update(values)

        # Generate dict with ncr input
        ncr = self.__prepare_labeling_input_ncr()

        # Retrieve defined labeling functions
        lfs = get_lfs(self.lf_input_dict, lf_input_ches, self.spacy_model)

        #L_gold_dev = load_gold_labels(session, annotator_name='gold', split=1)

        ## 2. Generate label matrix L
        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df=train_data)

        # Evaluate performance on training set
        print(LFAnalysis(L=L_train, lfs=lfs).lf_summary(Y=train_data.POPULIST.values))
        print(f"Training set coverage: {100 * LFAnalysis(L_train).label_coverage(): 0.1f}%")
        # print(f"Dev set coverage: {100 * LFAnalysis(L_dev).label_coverage(): 0.1f}%")

        analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary(Y=train_data.POPULIST.values)
        analysis.to_csv(f'{self.output_path}\\Snorkel\\snorkel_LF_analysis.csv')

        # Error analysis
        error_table = train_data.iloc[L_train[:, 1] == POP].sample(10, random_state=1)
        buckets = get_label_buckets(L_train[:, 0], L_train[:, 1])
        train_data.iloc[buckets[(ABSTAIN, POP)]].sample(10, random_state=1)

        #buckets = get_label_buckets(Y_gold, Y_pred)
        #buckets = get_label_buckets(train_data.POPULIST, L_train[:5])

        comparison_table = pd.DataFrame({'ID': train_data.ID,
                                        'text': train_data.text,
                                        'label': train_data.POPULIST,
                                         'lf_tfidf_global': L_train[:, 5]})


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

        ## 4. Train classifier
        # Filter out unlabeled data points
        probs_train = label_model.predict_proba(L=L_train)
        df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
            X=train_data, y=probs_train, L=L_train
        )

        vectorizer = CountVectorizer(ngram_range=(1, 5))
        #vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
        X_test = vectorizer.transform(test_data.text.tolist())

        preds_train_filtered = probs_to_preds(probs=probs_train_filtered)


        #sklearn_model = RandomForestClassifier()
        sklearn_model = LogisticRegression(C=1e3, solver="liblinear")
        sklearn_model.fit(X=X_train, y=preds_train_filtered)

        print(f"Test Accuracy: {sklearn_model.score(X=X_test, y=Y_test) * 100:.1f}%")

    def __prepare_labeling_input_ches(self):
        """
        Generate Lists of pop and nonpop parties for each CHES study
        :return: dict_ches_14, dict_ches_17, dict_ches_19
        :rtype:  dict
        """
        # Generate binary-labelled POP dataframes for Austrian, German and Swiss parties
        rel_countries = ['aus', 'ger', 'swi']
        rel_country_ids = [13, 3, 36]

        # CHES 14:
        ches_df_14 = pd.read_csv(f'{self.data_path}\\CHES\\2014_CHES_dataset_means.csv')
        ches_df_14 = ches_df_14.loc[ches_df_14.country.isin(rel_country_ids)]
        ches_df_14['antielite_salience'] = ches_df_14['antielite_salience'].apply(lambda x: 1 if x > 5 else 0)
        ches_df_14 = ches_df_14[['cname', 'party_name', 'antielite_salience']]
        # Rename "party_name" col
        ches_df_14.rename(columns={"party_name": "party", "cname": "country"}, inplace=True)

        # Split entries for parties with two naming conventions to separate entries
        add_entries_14 = pd.DataFrame()
        for index, row in ches_df_14.iterrows():
            # Find entries with two party namings (such as 'EDU/UDF')
            match = re.search(r'([^\/]+)\/([^\/]+)', row.party)
            if match:
                # Make first party the name of current row
                ches_df_14.loc[index, 'party'] = match.group(1)
                # Construct new row with second name and copy values for other cols
                entry = pd.DataFrame({'country': [row.country],
                                      'party': [match.group(2)],
                                      'antielite_salience': [row.antielite_salience]
                                      })
                # Add row to df which will be appended at the end
                add_entries_14 = add_entries_14.append(entry)

        # Add additional rows
        ches_df_14 = ches_df_14.append(add_entries_14)

        # Standardize party naming: lowercase, remove Umlaute, remove empty spaces, etc
        ches_df_14['party'] = ches_df_14['party'].apply(lambda x: standardize_party_naming(x))
        # Keep cdu/csu only once
        ches_df_14.drop_duplicates(keep='first', inplace=True)

        # Replace keys for countries
        ches_df_14['country'].replace({"ger": "de", "aus": "au", "swi": "cd"}, inplace=True)

        # Group by country
        ches_df_14_grpd = ches_df_14.groupby('country')

        dict_ches_14 = {}

        # Generate party list per country
        for index, row in ches_df_14_grpd:
            # Pop parties
            country_pop = row.loc[(row.antielite_salience == 1)]['party'].tolist()
            country_nonpop = row.loc[(row.antielite_salience == 0)]['party'].tolist()

            # Generate nested dict
            dict_ches_14[index] = {}
            dict_ches_14[index]['pop'] = country_pop
            dict_ches_14[index]['nonpop'] = country_nonpop

        # CHES 17:
        ches_df_17 = pd.read_csv(f'{self.data_path}\\CHES\\CHES_means_2017.csv')
        ches_df_17 = ches_df_17.loc[ches_df_17.country.isin(rel_countries)]
        ches_df_17['people_vs_elite'] = ches_df_17['people_vs_elite'].apply(lambda x: 1 if x > 5 else 0)
        ches_df_17['antielite_salience'] = ches_df_17['antielite_salience'].apply(lambda x: 1 if x > 5 else 0)
        ches_df_17 = ches_df_17[['country', 'party', 'people_vs_elite', 'antielite_salience']]

        # Split entries for parties with two naming conventions to separate entries
        add_entries_17 = pd.DataFrame()
        for index, row in ches_df_17.iterrows():
            # Find entries with two party namings (such as 'EDU/UDF')
            match = re.search(r'([^\/]+)\/([^\/]+)', row.party)
            if match:
                # Make first party the name of current row
                ches_df_17.loc[index, 'party'] = match.group(1)
                # Construct new row with second name and copy values for other cols
                entry = pd.DataFrame({'country': [row.country],
                                      'party': [match.group(2)],
                                      'people_vs_elite': [row.people_vs_elite],
                                      'antielite_salience': [row.antielite_salience]
                                      })
                # Add row to df which will be appended at the end
                add_entries_17 = add_entries_17.append(entry)

        # Add additional rows
        ches_df_17 = ches_df_17.append(add_entries_17)

        # Standardize party naming: lowercase, remove Umlaute, remove empty spaces, etc
        ches_df_17['party'] = ches_df_17['party'].apply(lambda x: standardize_party_naming(x))
        # Keep cdu/csu only once
        ches_df_17.drop_duplicates(keep='first', inplace=True)

        # Replace keys for countries
        ches_df_17['country'].replace({"ger": "de", "aus": "au", "swi": "cd"}, inplace=True)

        # Group by country
        ches_df_17_grpd = ches_df_17.groupby('country')

        dict_ches_17 = {}

        # Generate party list per country
        for index, row in ches_df_17_grpd:
            # Pop parties
            country_pop = row.loc[(row.people_vs_elite == 1) | (row.antielite_salience == 1)]['party'].tolist()
            country_nonpop = row.loc[(row.people_vs_elite == 0) & (row.antielite_salience == 0)]['party'].tolist()

            # Generate nested dict
            dict_ches_17[index] = {}
            dict_ches_17[index]['pop'] = country_pop
            dict_ches_17[index]['nonpop'] = country_nonpop

        # CHES 19:
        ches_df_19 = pd.read_csv(f'{self.data_path}\\CHES\\CHES2019V3.csv')
        ches_df_19 = ches_df_19.loc[ches_df_19.country.isin(rel_country_ids)]
        ches_df_19['people_vs_elite'] = ches_df_19['people_vs_elite'].apply(lambda x: 1 if x > 5 else 0)
        ches_df_19['antielite_salience'] = ches_df_19['antielite_salience'].apply(lambda x: 1 if x > 5 else 0)
        ches_df_19 = ches_df_19[['country', 'party', 'people_vs_elite', 'antielite_salience']]

        # Split entries for parties with two naming conventions to separate entries
        add_entries_19 = pd.DataFrame()
        for index, row in ches_df_19.iterrows():
            # Find entries with two party namings (such as 'EDU/UDF')
            match = re.search(r'([^\/]+)\/([^\/]+)', row.party)
            if match:
                # Make first party the name of current row
                ches_df_19.loc[index, 'party'] = match.group(1)
                # Construct new row with second name and copy values for other cols
                entry = pd.DataFrame({'country': [row.country],
                                      'party': [match.group(2)],
                                      'people_vs_elite': [row.people_vs_elite],
                                      'antielite_salience': [row.antielite_salience]
                                      })
                # Add row to df which will be appended at the end
                add_entries_19 = add_entries_19.append(entry)

        # Add additional rows
        ches_df_19 = ches_df_19.append(add_entries_19)

        # Standardize party naming: lowercase, remove Umlaute, remove empty spaces, etc
        ches_df_19['party'] = ches_df_19['party'].apply(lambda x: standardize_party_naming(x))
        # Keep cdu/csu only once
        ches_df_19.drop_duplicates(keep='first', inplace=True)

        # Replace country_ids with country names
        ches_df_19['country'].replace({3: "de", 13: "au", 36: "cd"}, inplace=True)

        # Group by country
        ches_df_19_grpd = ches_df_19.groupby('country')

        dict_ches_19 = {}

        # Generate party list per country
        for index, row in ches_df_19_grpd:
            # Pop parties
            country_pop = row.loc[(row.people_vs_elite == 1) | (row.antielite_salience == 1)]['party'].tolist()
            country_nonpop = row.loc[(row.people_vs_elite == 0) & (row.antielite_salience == 0)]['party'].tolist()

            # Generate nested dict
            dict_ches_19[index] = {}
            dict_ches_19[index]['pop'] = country_pop
            dict_ches_19[index]['nonpop'] = country_nonpop

        return dict_ches_14, dict_ches_17, dict_ches_19

    def __prepare_labeling_input_ncr(self):
        """
        Generate dataframe for ncr sentiment analysis
        :return: ncr
        :rtype:  DataFrame
        """
        ncr = pd.read_csv(f'{self.data_path}\\NRC-Emotion-Lexicon\\NRC-Emotion-Lexicon\\NRC-Emotion-Lexicon-v0.92\\'
                          f'NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.csv', sep=';')

        # todo: generate sentiment from ncr
        # Subselect relevant columns
        ncr = ncr[['German (de)', 'Positive', 'Negative',
                   'Anger', 'Anticipation', 'Disgust',
                   'Fear', 'Joy', 'Sadness', 'Surprise',
                   'Trust']]

        return ncr
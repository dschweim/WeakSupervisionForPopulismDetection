# This main file contains function calls for all other python files
# In order to avoid long runtime set parameters to false after the data output has been initially created
import pandas as pd
import spacy
import sys
from argparse import ArgumentParser
from NCCR_Corpus import PCCR_Dataset
from Labeling_Framework import snorkel_labeling
from util import generate_train_test_split, generate_train_dev_test_split
import spacy
import torch
import numpy
from numpy.testing import assert_almost_equal


sys.path.append("..")


def main(input_path, generate_data, run_labeling, generate_train_test, generate_tfidf_dicts):
    # Set project path globally
    path_to_project_folder = input_path

    # Initialize
    df_nccr = PCCR_Dataset(data_path=f'{path_to_project_folder}\\Data',
                           output_path=f'{path_to_project_folder}\\Output')

    # Either generate data or read data from disk
    if generate_data:
        # Generate Labelled NCCR
        nccr_data_de_wording_all, nccr_data_de_wording_av = df_nccr.generate_labelled_nccr_corpus()

    else:
        # Import corpora
        nccr_data_de_wording_all = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_all.csv'
        )
        nccr_data_de_wording_av = pd.read_csv(
            f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_available.csv'
        )

    # Run Snorkel framework if set
    if run_labeling:

        # Either generate train test split or read from disk
        if generate_train_test:
            # Generate Train, Test Split
            train, test = generate_train_test_split(nccr_data_de_wording_av)
            # Pre-process data
            train_prep = df_nccr.preprocess_corpus(train, is_train=True)
            test_prep = df_nccr.preprocess_corpus(test, is_train=False)

            # Generate Train, Dev, Test Split
            #train, dev, test = generate_train_dev_test_split(nccr_data_de_wording_av)

        else:
            # Import preprocessed data
            train_prep = pd.read_csv(
                f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_available_TRAIN.csv'
            )
            test_prep = pd.read_csv(
                f'{path_to_project_folder}\\Output\\NCCR_combined_corpus_DE_wording_available_TEST.csv'
            )

        if generate_tfidf_dicts:
            # Generate Dictionaries based on tfidf
            tfidf_dict = df_nccr.generate_tfidf_dict(train_prep, tfidf_threshold=30)
            tfidf_dict_country = df_nccr.generate_tfidf_dict_per_country(train_prep, tfidf_threshold=30)
            tfidf_dict_global = df_nccr.generate_global_tfidf_dict(train_prep, tfidf_threshold=30)

        else:
            # Import dictionaries
            tfidf_dict = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict.csv'
            )

            tfidf_dict_country_au = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict_per_country_au.csv'
            )
            tfidf_dict_country_ch = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict_per_country_ch.csv'
            )
            tfidf_dict_country_de = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict_per_country_de.csv'
            )

            tfidf_dict_country = {}
            values = {'au': tfidf_dict_country_au,
                      'cd': tfidf_dict_country_ch,
                      'de': tfidf_dict_country_de}
            tfidf_dict_country.update(values)

            tfidf_dict_global = pd.read_csv(
                f'{path_to_project_folder}\\Output\\tfidf_dict_global.csv'
            )


        # Generate overall dictionary as labeling function input
        lf_dict = {'tfidf_keywords': tfidf_dict.term.to_list(),
                   'tfidf_keywords_at': tfidf_dict_country['au'].term.to_list(),
                   'tfidf_keywords_ch': tfidf_dict_country['cd'].term.to_list(),
                   'tfidf_keywords_de': tfidf_dict_country['de'].term.to_list(),
                   'tfidf_keywords_global': tfidf_dict_global.term.to_list()}

        # Filter on relevant columns
        train_prep_sub = train_prep[['text_prep', 'party', 'Sample_Country', 'year', 'POPULIST']]
        test_prep_sub = test_prep[['text_prep', 'party', 'Sample_Country', 'year', 'POPULIST']]
        train_prep_sub.rename({'text_prep': 'text'}, axis=1, inplace=True)
        test_prep_sub.rename({'text_prep': 'text'}, axis=1, inplace=True)

        # todo: identify people embedding
        # todo: Spacy preprocessing

        nlp_trf = spacy.load("de_dep_news_trf")

        # Import the Language object under the 'language' module in spaCy,
        # and NumPy for calculating cosine similarity.
        from spacy.language import Language
        import numpy as np

        # We use the @ character to register the following Class definition
        # with spaCy under the name 'tensor2attr'.
        @Language.factory('tensor2attr')

        # We begin by declaring the class name: Tensor2Attr. The name is
        # declared using 'class', followed by the name and a colon.
        class Tensor2Attr:

            # We continue by defining the first method of the class,
            # __init__(), which is called when this class is used for
            # creating a Python object. Custom components in spaCy
            # require passing two variables to the __init__() method:
            # 'name' and 'nlp'. The variable 'self' refers to any
            # object created using this class!
            def __init__(self, name, nlp):
                # We do not really do anything with this class, so we
                # simply move on using 'pass' when the object is created.
                pass

            # The __call__() method is called whenever some other object
            # is passed to an object representing this class. Since we know
            # that the class is a part of the spaCy pipeline, we already know
            # that it will receive Doc objects from the preceding layers.
            # We use the variable 'doc' to refer to any object received.
            def __call__(self, doc):
                # When an object is received, the class will instantly pass
                # the object forward to the 'add_attributes' method. The
                # reference to self informs Python that the method belongs
                # to this class.
                self.add_attributes(doc)

                # After the 'add_attributes' method finishes, the __call__
                # method returns the object.
                return doc

            # Next, we define the 'add_attributes' method that will modify
            # the incoming Doc object by calling a series of methods.
            def add_attributes(self, doc):
                # spaCy Doc objects have an attribute named 'user_hooks',
                # which allows customising the default attributes of a
                # Doc object, such as 'vector'. We use the 'user_hooks'
                # attribute to replace the attribute 'vector' with the
                # Transformer output, which is retrieved using the
                # 'doc_tensor' method defined below.
                doc.user_hooks['vector'] = self.doc_tensor

                # We then perform the same for both Spans and Tokens that
                # are contained within the Doc object.
                doc.user_span_hooks['vector'] = self.span_tensor
                doc.user_token_hooks['vector'] = self.token_tensor

                # We also replace the 'similarity' method, because the
                # default 'similarity' method looks at the default 'vector'
                # attribute, which is empty! We must first replace the
                # vectors using the 'user_hooks' attribute.
                doc.user_hooks['similarity'] = self.get_similarity
                doc.user_span_hooks['similarity'] = self.get_similarity
                doc.user_token_hooks['similarity'] = self.get_similarity

            # Define a method that takes a Doc object as input and returns
            # Transformer output for the entire Doc.
            def doc_tensor(self, doc):
                # Return Transformer output for the entire Doc. As noted
                # above, this is the last item under the attribute 'tensor'.
                # Average the output along axis 0 to handle batched outputs.
                return doc._.trf_data.tensors[-1].mean(axis=0)

            # Define a method that takes a Span as input and returns the Transformer
            # output.
            def span_tensor(self, span):
                # Get alignment information for Span. This is achieved by using
                # the 'doc' attribute of Span that refers to the Doc that contains
                # this Span. We then use the 'start' and 'end' attributes of a Span
                # to retrieve the alignment information. Finally, we flatten the
                # resulting array to use it for indexing.
                tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()

                # Fetch Transformer output shape from the final dimension of the output.
                # We do this here to maintain compatibility with different Transformers,
                # which may output tensors of different shape.
                out_dim = span.doc._.trf_data.tensors[0].shape[-1]

                # Get Token tensors under tensors[0]. Reshape batched outputs so that
                # each "row" in the matrix corresponds to a single token. This is needed
                # for matching alignment information under 'tensor_ix' to the Transformer
                # output.
                tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

                # Average vectors along axis 0 ("columns"). This yields a 768-dimensional
                # vector for each spaCy Span.
                return tensor.mean(axis=0)

            # Define a function that takes a Token as input and returns the Transformer
            # output.
            def token_tensor(self, token):
                # Get alignment information for Token; flatten array for indexing.
                # Again, we use the 'doc' attribute of a Token to get the parent Doc,
                # which contains the Transformer output.
                tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()

                # Fetch Transformer output shape from the final dimension of the output.
                # We do this here to maintain compatibility with different Transformers,
                # which may output tensors of different shape.
                out_dim = token.doc._.trf_data.tensors[0].shape[-1]

                # Get Token tensors under tensors[0]. Reshape batched outputs so that
                # each "row" in the matrix corresponds to a single token. This is needed
                # for matching alignment information under 'tensor_ix' to the Transformer
                # output.
                tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

                # Average vectors along axis 0 (columns). This yields a 768-dimensional
                # vector for each spaCy Token.
                return tensor.mean(axis=0)

            # Define a function for calculating cosine similarity between vectors
            def get_similarity(self, doc1, doc2):
                # Calculate and return cosine similarity
                return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)

        nlp_trf.add_pipe('tensor2attr')

        # Call the 'pipeline' attribute to examine the pipeline

        key_word = nlp_trf("Regierung")[0]
        texts = [train_prep.text_prep[100], train_prep.text_prep[10]]

        # compute similarity between keyword and each word of each doc

        comparison = [train_prep.text_prep[10], train_prep.text_prep[100]]
        similarities = {}

        for doc in nlp_trf.pipe(comparison):
            similarities[doc.text] = {}
            for span in doc.char_span:
                for token in span:
                    print(token.i)
                    print(token.text)
                    sim = doc[token.i].similarity(key_word)
                    similarities[doc.text].update({token.text: sim})




        # Run Snorkel framework
        snorkel_labeling(train_data=train_prep_sub,
                         test_data=test_prep_sub,
                         lf_input_dict=lf_dict,
                         data_path=f'{path_to_project_folder}\\Data',
                         output_path=f'{path_to_project_folder}\\Output')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input

    main(input_path=input_path,
         generate_data=False,
         run_labeling=True,
         generate_train_test=False,
         generate_tfidf_dicts=False)

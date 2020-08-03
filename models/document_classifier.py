from typing import Tuple, Union, List, Optional
import os
import logging
from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from loader import load_corpus_as_dataframe, load_translations_to_df
from preprocess.normalize import normalize_document
from preprocess.doc_embedding import (
    document_to_vector,
    words_in_model,
    load_word_embeddings_model,
    DummyEmbedding,
)
from preprocess.clean import clean_text
from preprocess.detect_language import detected_language
from translation.translate import translate_to_spanish_from_df
from custom_types import Prediction, TranslationInput
from utils import log, fix_seeds


fix_seeds()
tqdm.pandas()


class DocumentClassifier(object):
    """docstring for DocumentClassifier"""

    def __init__(self):
        super(DocumentClassifier, self).__init__()
        self.corpus_dir: str = os.path.join("data", "documents_challenge")
        self.translations_dir: str = os.path.join("data", "translations_es")
        self.class_encoder = self.class_decoder = None
        self._load_models()

    def _load_models(self):
        log.info("Loading embeddings model")
        # self.embeddings_model = DummyEmbedding()
        self.embeddings_model = load_word_embeddings_model()

    def _load_data(self) -> pd.DataFrame:
        df = load_corpus_as_dataframe(self.corpus_dir)
        log.info(f"Loaded {len(df)} samples")

        df = load_translations_to_df(df, self.translations_dir)
        df = df[~df.translated_text.isnull()]
        log.info(f"{len(df)} translated documents")

        return df

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Preprocess
        log.info("Normalizing documents' text")
        df["normalized_text"] = df.translated_text.progress_apply(
            lambda x: normalize_document(x, self.embeddings_model)
        )

        log.info("Calculating document embeddings")

        df["document_embeddings"] = df.normalized_text.progress_apply(
            lambda x: document_to_vector(
                words_in_model(x, self.embeddings_model), self.embeddings_model
            )
        )

        df = df[~df.document_embeddings.isnull()]
        return df

    def _calculate_class_encoder(self, categories_series: pd.Series):
        self.class_encoder = {
            k: v
            for k, v in zip(
                categories_series.values.tolist(), categories_series.cat.codes.tolist()
            )
        }

        self.class_decoder = {v: k for k, v in self.class_encoder.items()}

    def _split_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(df, test_size=0.1)

        return train_df, test_df

    @staticmethod
    def _embeddings_list_to_dmatrix(
        embeddings_list: List[np.array], label: Optional[np.array] = None
    ) -> xgb.DMatrix:
        stacked_embeddings = np.stack(embeddings_list)

        dmatrix = (
            xgb.DMatrix(stacked_embeddings, label=label,)
            if label is not None
            else xgb.DMatrix(stacked_embeddings)
        )

        return dmatrix

    def _train_xgboot(self, train_df: pd.DataFrame, test_df: pd.DataFrame):

        y_true = test_df.context_categorical.cat.codes

        dtrain = self._embeddings_list_to_dmatrix(
            train_df.document_embeddings.tolist(),
            label=train_df.context_categorical.cat.codes,
        )
        dtest = self._embeddings_list_to_dmatrix(
            test_df.document_embeddings.tolist(), label=y_true
        )

        num_class = max(train_df.context_categorical.cat.codes) + 1

        # Fit model
        param = {
            "max_depth": 2,
            "eta": 1,
            "objective": "multi:softprob",  # "binary:logistic",
            "num_class": num_class,
            "nthread": 4,
        }

        evallist = [(dtrain, "train"), (dtest, "test")]
        num_round = 20
        self.bst = xgb.train(param, dtrain, num_round, evallist)

        # Evaluate
        pred_probs = self.bst.predict(dtest)
        y_pred = np.argmax(pred_probs, axis=1)

        print(classification_report(y_true, y_pred))

    def train(self):
        df = self._load_data()
        df = self._preprocess_df(df)
        df["context_categorical"] = pd.Categorical(df.context)
        self._calculate_class_encoder(df.context_categorical)
        self.train_df, self.test_df = self._split_df(df)
        self._train_xgboot(self.train_df, self.test_df)
        self.save()

    def evaluate(self):
        y_true = self.test_df.context_categorical.cat.codes
        dtest = self._embeddings_list_to_dmatrix(
            self.test_df.document_embeddings.tolist(), label=y_true
        )

        pred_probs = self.bst.predict(dtest)
        y_pred = np.argmax(pred_probs, axis=1)

        print(classification_report(y_true, y_pred))

    def predict(self, documents: TranslationInput) -> List[Prediction]:

        if isinstance(documents, str):
            documents = [documents]

        df = pd.DataFrame({"text": documents}, columns=["text"])

        # Let's clean the text and detect the language before preprocessing
        df = clean_text(df)
        df = detected_language(df)
        df = translate_to_spanish_from_df(df)

        df = self._preprocess_df(df)

        dpredict = xgb.DMatrix(np.stack(df.document_embeddings.tolist()))
        pred_probs = self.bst.predict(dpredict)

        y_pred = np.argmax(pred_probs, axis=1)
        prob_pred = np.take_along_axis(
            pred_probs, np.expand_dims(y_pred, axis=1), axis=1
        )

        predictions = [
            Prediction(language, self.class_decoder[y], prob[0])
            for language, y, prob in zip(
                df.detected_language.tolist(), y_pred.tolist(), prob_pred.tolist()
            )
        ]

        return predictions

    def save(self, filename: str = "document_classifier.joblib.pkl"):
        joblib.dump(self, filename)

    @classmethod
    def load(
        self, filename: str = "document_classifier.joblib.pkl"
    ) -> "DocumentClassifier":
        log.warning(f"Loading pretrained DocumentClassifier from {filename}")
        instance = joblib.load(filename)
        return instance


if __name__ == "__main__":
    # Train the model yourself
    # classifier = DocumentClassifier()
    # classifier.train()

    # Load the model from a pickle
    classifier = DocumentClassifier.load()
    classifier.evaluate()
    breakpoint()

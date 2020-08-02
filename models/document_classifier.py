from typing import Tuple, Union
import os
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
from custom_types import Prediction, TranslationInput
from utils import log


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
        df = df[~df.loaded_translation.isnull()]
        log.info(f"{len(df)} translated documents")

        return df

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Preprocess
        log.info("Normalizing documents' text")
        df["normalized_text"] = df.loaded_translation.progress_apply(
            lambda x: normalize_document(x, self.embeddings_model)
        )

        log.info("Calculating document embeddings")

        df["document_embeddings"] = df.normalized_text.progress_apply(
            lambda x: document_to_vector(
                words_in_model(x, self.embeddings_model), self.embeddings_model
            )
        )

        df = df[~df.document_embeddings.isnull()]

        df["context_categorical"] = pd.Categorical(df.context)
        return df

    def _calculate_class_encoder(self, categories_series: pd.Series):
        self.class_encoder = {
            k: v for k, v in
            zip(categories_series.cat.codes, categories_series.cat.categories)
        }

        self.class_decoder = {v: k for k, v in self.class_encoder.items()}

    def _split_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(df, test_size=0.1)

        return train_df, test_df

    def _train_xgboot(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        train_stacked_document_embeddings = np.stack(
            train_df.document_embeddings.tolist()
        )
        test_stacked_document_embeddings = np.stack(
            test_df.document_embeddings.tolist()
        )

        y_true = test_df.context_categorical.cat.codes

        # Define model
        dtrain = xgb.DMatrix(
            train_stacked_document_embeddings,
            label=train_df.context_categorical.cat.codes,
        )
        dtest = xgb.DMatrix(test_stacked_document_embeddings, label=y_true)

        num_class = max(train_df.context_categorical.cat.codes) + 1

        # Fit model
        param = {
            "max_depth": 2,
            "eta": 1,
            "objective": "multi:softprob",  # "binary:logistic",
            "num_class": num_class,
        }
        param["nthread"] = 4
        # param['eval_metric'] = 'auc'
        evallist = [(dtrain, "eval"), (dtrain, "train")]
        num_round = 10
        self.bst = xgb.train(param, dtrain, num_round, evallist)

        # Evaluate
        pred_probs = self.bst.predict(dtest)
        y_pred = np.argmax(pred_probs, axis=1)

        print(classification_report(y_true, y_pred))

    def train(self):
        df = self._load_data()
        # FIXME
        # df = df.sample(n=1000)
        df = self._preprocess_df(df)
        self._calculate_class_encoder(df.context_categorical)
        train_df, test_df = self._split_df(df)
        self._train_xgboot(train_df, test_df)

    def predict(self, documents: TranslationInput) -> Prediction:

        if isinstance(documents, str):
            documents = [documents]

        df = pd.DataFrame({"text": documents}, columns=["text"])

        # Let's clean the text and detect the language before preprocessing
        df = clean_text(df)
        df = detected_language(df)

        df = self._preprocess_df(df)

        dpredict = xgb.DMatrix(np.stack(
            df.document_embeddings.tolist()
        ))
        pred_probs = self.bst.predict(dpredict)

        y_pred = np.argmax(pred_probs, axis=1)
        prob_pred = pred_probs[y_pred]

        return Prediction(self.class_decoder[y_pred], prob_pred)

if __name__ == "__main__":
    classifier = DocumentClassifier()
    classifier.train()
    breakpoint()

from typing import Tuple, List, Optional
from pprint import pprint

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from preprocess.clean import clean_text
from preprocess.detect_language import detected_language
from translation.translate import translate_to_spanish_from_df
from custom_types import Prediction, TranslationInput
from utils import parse_execution_mode

from .base_model import BaseModel


class DocumentClassifier(BaseModel):
    """docstring for DocumentClassifier"""

    def __init__(self):
        super(DocumentClassifier, self).__init__()
        self.class_encoder = self.class_decoder = None
        self._load_embeddings_model()

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_df(df, self.embeddings_model)
        df = self._documents_to_embedding_vector_df(df)
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

        num_class = max(train_df.context_categorical.cat.codes) + 1
        y_true = test_df.context_categorical.cat.codes

        dtrain = self._embeddings_list_to_dmatrix(
            train_df.document_embeddings.tolist(),
            label=train_df.context_categorical.cat.codes,
        )
        dtest = self._embeddings_list_to_dmatrix(
            test_df.document_embeddings.tolist(), label=y_true
        )

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


if __name__ == "__main__":
    execution_mode = parse_execution_mode()

    if execution_mode == "train":
        # Train the model yourself
        classifier = DocumentClassifier()
        classifier.train()
        classifier.save()
    elif execution_mode == "evaluate":
        # Load the model from a pickle
        classifier = DocumentClassifier.load()
        classifier.evaluate()
    elif execution_mode == "predict":
        # Interactive predictions mode!
        classifier = DocumentClassifier.load()
        while True:
            try:
                sentence = input("Enter a sentence to classify: ")
                prediction = classifier.predict(sentence)[0]
                pprint(prediction)
            except KeyboardInterrupt:
                break
    else:
        raise ValueError(
            f"Execution mode {execution_mode} not implemented for this module"
        )

from typing import Tuple, Union, List
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
            Prediction(self.class_decoder[y], prob[0])
            for y, prob in zip(y_pred.tolist(), prob_pred.tolist())
        ]

        breakpoint()
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
    prediction = classifier.predict(
        [
            """
Elle se situe au cœur d'un vaste bassin sédimentaire aux sols fertiles et au climat tempéré, le bassin parisien, sur une boucle de la Seine, entre les confluents de celle-ci avec la Marne et l'Oise. Paris est également le chef-lieu de la région Île-de-France et le centre de la métropole du Grand Paris, créée en 2016. Elle est divisée en arrondissements, comme les villes de Lyon et de Marseille, au nombre de vingt. Administrativement, la ville constitue depuis le 1er janvier 2019 une collectivité à statut particulier nommée « Ville de Paris » (auparavant, elle était à la fois une commune et un département). L'État y dispose de prérogatives particulières exercées par le préfet de police de Paris. La ville a connu de profondes transformations sous le Second Empire dans les décennies 1850 et 1860 à travers d'importants travaux consistant notamment au percement de larges avenues, places et jardins et la construction de nombreux édifices, dirigés par le baron Haussmann, donnant à l'ancien Paris médiéval le visage qu'on lui connait aujourd'hui.
La ville de Paris comptait 2,187 millions d'habitants au 1er janvier 2020. Ses habitants sont appelés Parisiens. L'agglomération parisienne s’est largement développée au cours du xxe siècle, rassemblant 10,73 millions d'habitants au 1er janvier 2020, et son aire urbaine (l'agglomération et la couronne périurbaine) comptait 12,78 millions d'habitants. L'agglomération parisienne est ainsi la plus peuplée de France, elle est la quatrième du continent européen et la 32e plus peuplée du monde au 1er janvier 2019
""",
            """i read this book because in my town, everyone uses it and order. this is my pharmacist who advised me she was so thin i asked her what she had done and instead of just selling snake oil capsules, she advised me this book to 5 euros. of course, we must make an effort to lose 25 pounds but with the book, i had a companion. the author was able to talk to me just with strong arguments and above all i felt he knew many cases like mine. he is in his full experience, simplicity and compassion for those like me who lived with all that weight that stuck to my body and never want to leave. i do not think it is a fad diet that outperforms the others but i do believe that there are people who can speak to others and to be born of clicks. i might be low but this book made me strong, i have annotated so that i'm on my third. when one is very big as i was, non-large do not understand or are afraid to offend you by speaking, then this book was like a companion journal. i am a pedicure and i have advised all my clients that i read great suffering on their feet swollen and deformed. i can provide other service that i made my pharmacist. i advise all those who suffer for having lost weight is such a happiness that i agreed to move to phase 3 of this plan, which requires 10 days of consolidation for each kilo lost gradually widening at all. now i'm in stage 4, meaning that i eat everything except on thursdays when i control. i never thank enough the author of this book.""",
        ]
    )
    pprint(prediction)
    breakpoint()

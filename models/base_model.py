import os

from tqdm import tqdm
import joblib
import pandas as pd
import spacy
from nltk.stem import SnowballStemmer

from loader import load_corpus_as_dataframe, load_translations_to_df
from preprocess.doc_embedding import (
    document_to_vector,
    words_in_model,
    load_word_embeddings_model,
    # DummyEmbedding,
)
from preprocess.normalize import normalize_document
from utils import log, fix_seeds

fix_seeds()
tqdm.pandas()

STEMMER = SnowballStemmer("spanish")


class BaseModel(object):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.corpus_dir: str = os.path.join("data", "documents_challenge")
        self.translations_dir: str = os.path.join("data", "translations_es")
        self.nlp = spacy.load("es", disable=["parser", "ner"])

    @classmethod
    def _default_pickle_name(cls) -> str:
        return f"data/{cls.__name__}.joblib.pkl"

    def save(self, filename: str = None):
        filename = filename or self._default_pickle_name()
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename: str = None) -> "BaseModel":
        filename = filename or cls._default_pickle_name()
        log.warning(f"Loading pretrained {cls.__class__.__name__} from {filename}")
        instance = joblib.load(filename)
        return instance

    def _load_data(self) -> pd.DataFrame:
        df = load_corpus_as_dataframe(self.corpus_dir)
        log.info(f"Loaded {len(df)} samples")

        # df = df.sample(n=500)  # FIXME

        df = load_translations_to_df(df, self.translations_dir)
        df = df[~df.translated_text.isnull()]
        log.info(f"{len(df)} translated documents")
        return df

    def _load_embeddings_model(self):
        log.info("Loading embeddings model")
        # self.embeddings_model = DummyEmbedding()
        self.embeddings_model = load_word_embeddings_model()

    def _normalize_df(self, df: pd.DataFrame, embeddings_model=None) -> pd.DataFrame:
        log.info("Normalizing documents")
        df["normalized_text"] = df.translated_text.progress_apply(
            lambda x: normalize_document(x, embeddings_model)
        )
        return df

    def _documents_to_embedding_vector_df(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Calculating document embeddings")
        df["document_embeddings"] = df.normalized_text.progress_apply(
            lambda x: document_to_vector(
                words_in_model(x, self.embeddings_model), self.embeddings_model
            )
        )
        df = df[~df.document_embeddings.isnull()]
        return df

    def _lemmatize_df(
        self,
        df: pd.DataFrame,
        allowed_postags=["NOUN", "ADJ"],  # ["NOUN", "ADJ", "VERB", "ADV"]
    ) -> pd.DataFrame:
        def lemmatize_sent(sent):
            doc = self.nlp(" ".join(sent))
            return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

        df["lemmatized_text"] = df.normalized_text.progress_apply(lemmatize_sent)
        df = df[df.lemmatized_text.apply(lambda x: len(x)) > 0]
        df["stemmed_text"] = df.lemmatized_text.progress_apply(
            lambda x: [STEMMER.stem(token) for token in x]
        )
        return df

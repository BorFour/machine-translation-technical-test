"""

https://github.com/dccuchile/spanish-word-embeddings
https://medium.com/@paritosh_30025/natural-language-processing-text-data-vectorization-af2520529cf7
"""
from typing import List, Union

import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import numpy as np

from utils import log

nltk.download("punkt")


class DummyEmbedding(KeyedVectors):
    """The real FastText embeddings used for the document classifier take quite long to load.
    Thus, this class implements the basic behavior needed to emulate an embedding model with dummy methods"""

    def __init__(self):
        pass

    def __getitem__(self, key):
        return np.zeros((300,))

    def __contains__(self, element):
        return True


def load_word_embeddings_model(path: str = "data/fasttext-sbwc.vec.gz") -> KeyedVectors:
    log.info(f"Loading model from {path}")
    model = KeyedVectors.load_word2vec_format(path)

    return model


def words_in_model(document: List[str], model: KeyedVectors) -> List[str]:
    return [word for word in document if word.lower() in model]


def document_to_vector(
    document: List[str], model: KeyedVectors
) -> Union[np.array, None]:

    if not document:
        return None

    embeddings = []
    for word in document:
        if word.lower() in model:
            embeddings.append(model[word.lower()])

    stacked_embeddings = np.stack(embeddings)
    doc_embedding = stacked_embeddings.mean(axis=0)

    return doc_embedding


if __name__ == "__main__":
    model = load_word_embeddings_model()
    sentence = "Mi gato se llama Guantes"
    tokens = word_tokenize(sentence)
    tokens_in_model = words_in_model(tokens, model)
    v = document_to_vector(tokens_in_model, model)

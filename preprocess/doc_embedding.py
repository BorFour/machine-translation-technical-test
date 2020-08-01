"""

https://github.com/dccuchile/spanish-word-embeddings
https://medium.com/@paritosh_30025/natural-language-processing-text-data-vectorization-af2520529cf7
"""
from typing import List, Tuple

import nltk
from nltk.tokenize import word_tokenize
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from gensim import corpora
import numpy as np

from utils import log

nltk.download("punkt")

## Exapmple document (list of sentences)
texts = [
    "I love data science",
    "I love coding in python",
    "I love building NLP tool",
    "This is a good phone",
    "This is a good TV",
    "This is a good laptop",
]


def load_word_embeddings_model(path: str = "data/fasttext-sbwc.vec.gz") -> KeyedVectors:
    # model = FastText.load_fasttext_format('data/embeddings-l-model')
    log.info(f"Loading model from {path}")
    model = KeyedVectors.load_word2vec_format(path)

    return model


def words_in_model(document: List[str], model: KeyedVectors) -> List[str]:
    return [word for word in document if word.lower() in model]


def document_to_vector(
    document: List[str], model: KeyedVectors
) -> Tuple[List[str], np.array]:

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
    breakpoint()

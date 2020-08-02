import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from custom_types import Document, TokenizedDocument, Vocabulary


nltk.download("stopwords")
SPANISH_STOPWORDS = set(stopwords.words("spanish"))


def tokenize_document(document: Document) -> TokenizedDocument:
    return word_tokenize(document)


def remove_stopwords(tokens: TokenizedDocument) -> TokenizedDocument:
    global SPANISH_STOPWORDS
    return [token for token in tokens if token.lower() not in SPANISH_STOPWORDS]


def normalize_document(document: Document, vocab: Vocabulary) -> TokenizedDocument:
    tokenized_document = tokenize_document(document)
    tokenized_document = [token.lower() for token in tokenized_document]
    unique_tokens = list(set(tokenized_document))
    tokens = remove_stopwords(unique_tokens)
    tokens_in_vocabulary = [t for t in tokens if t in vocab]

    return tokens_in_vocabulary

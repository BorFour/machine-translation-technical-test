import os

import pandas as pd
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora
import spacy

from visualize.wordcloud import topics_wordcloud
from utils import parse_execution_mode

from .base_model import BaseModel


class LdaTopicModelling(BaseModel):
    """docstring for LdaTopicModelling"""

    def __init__(self, num_topics: int = 10):
        super(LdaTopicModelling, self).__init__()
        self.num_topics = num_topics
        self.corpus_dir: str = os.path.join("data", "documents_challenge")
        self.translations_dir: str = os.path.join("data", "translations_es")
        self.nlp = spacy.load("es", disable=["parser", "ner"])

    def _train_lda(self, df: pd.DataFrame):
        self.docs = df.lemmatized_text.tolist()
        self.dictionary = corpora.Dictionary(self.docs)
        # self.dictionary.filter_extremes(no_below=50, no_above=0.5)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.docs]

        print("Number of unique tokens: %d" % len(self.dictionary))
        print("Number of documents: %d" % len(self.corpus))

        self.lda = LdaMulticore(
            self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=20
        )

    def visualize_lda(self):
        import pyLDAvis.gensim

        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(
            self.lda, self.corpus, dictionary=self.lda.id2word
        )
        return vis

    def train(self):
        df = self._load_data()
        df = self._normalize_df(df)
        df = self._lemmatize_df(df)
        self._train_lda(df)


if __name__ == "__main__":
    execution_mode = parse_execution_mode()

    if execution_mode == "train":
        topic_modelling = LdaTopicModelling(num_topics=6)
        topic_modelling.train()
        topic_modelling.save()
    elif execution_mode == "evaluate":
        topic_modelling = LdaTopicModelling.load()
        topics_wordcloud(
            topic_modelling.lda.show_topics(formatted=False),
            f"imgs/lda_topics_{topic_modelling.num_topics}.png",
        )
    else:
        raise ValueError(
            f"Execution mode {execution_mode} not implemented for this module"
        )

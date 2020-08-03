from collections import OrderedDict

import numpy as np
from sklearn.cluster import KMeans

from visualize.wordcloud import topics_wordcloud
from utils import parse_execution_mode

from .base_model import BaseModel


class KMeansTopicModelling(BaseModel):
    """docstring for KMeansTopicModelling"""

    def __init__(self, num_topics: int = 10, words_per_centroid: int = 50):
        super(KMeansTopicModelling, self).__init__()
        self.num_topics = num_topics
        self.words_per_centroid = words_per_centroid
        self._load_embeddings_model()

    def get_topic_words(self):
        self.topic_words = OrderedDict()
        for i in range(self.kmeans.cluster_centers_.shape[0]):
            centroid = self.kmeans.cluster_centers_[i]
            centroid_similar_words = self.embeddings_model.similar_by_vector(
                centroid, topn=self.words_per_centroid
            )
            self.topic_words[i] = centroid_similar_words

    def train(self):
        df = self._load_data()
        df = self._normalize_df(df, self.embeddings_model)
        df = self._lemmatize_df(df)
        df = self._documents_to_embedding_vector_df(df)

        breakpoint()

        self.X = np.stack(df.document_embeddings)
        self.kmeans = KMeans(self.num_topics)
        self.kmeans.fit(self.X)
        self.get_topic_words()


if __name__ == "__main__":
    execution_mode = parse_execution_mode()

    if execution_mode == "train":
        topic_modelling = KMeansTopicModelling(num_topics=6)
        topic_modelling.train()
        topic_modelling.save()
    elif execution_mode == "evaluate":
        topic_modelling = KMeansTopicModelling.load()
        topics_wordcloud(
            [
                (cluster_id, words)
                for cluster_id, words in topic_modelling.topic_words.items()
            ],
            f"imgs/kmeans_topics_{topic_modelling.num_topics}.png",
        )
    else:
        raise ValueError(
            f"Execution mode {execution_mode} not implemented for this module"
        )

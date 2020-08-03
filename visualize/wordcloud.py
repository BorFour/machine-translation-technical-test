# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import math


def topics_wordcloud(topics, output_filename: str = None):

    n_topics = len(topics)
    n_rows = math.ceil(n_topics / 2)
    n_columns = 2

    cols = [
        color for name, color in mcolors.TABLEAU_COLORS.items()
    ]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(
        # stopwords=stop_words,
        background_color="white",
        width=1200,
        height=400 * n_rows,
        max_words=10,
        colormap="tab10",
        color_func=lambda *args, **kwargs: cols[i],
        prefer_horizontal=1.0,
    )

    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(10, 10), sharex=True, sharey=True
    )

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title("Topic " + str(i), fontdict=dict(size=16))
        plt.gca().axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename)

    plt.show()

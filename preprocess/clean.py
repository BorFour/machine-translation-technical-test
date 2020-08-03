import re

import pandas as pd

from loader import load_corpus_as_dataframe, DOCUMENTS_DIR


TRAILING_STUFF_WIKIPEDIA = re.compile(r"(Categoría|Catégorie|Category):.+$")
OTHER_LANGUAGES_REFERNECE_WIKIPEDIA = re.compile(r"\*\s[a-z]+:.*$")


def clean_pixels(text: str) -> str:
    return re.sub(r"[0-9]+px\|?", "", text)


def clean_box_assign(text: str) -> str:
    return re.sub(r"\w+\s?\=\s?\w*|\|", "", text)


def remove_long_words(text: str) -> str:
    return re.sub(r"\w{40,}", "", text)


def remove_urls(text: str) -> str:
    return re.sub(
        r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
        "",
        text,
    )


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Remove some unmeaningful characters, so it makes the translations easier."""

    df["cleaned_text"] = (
        df.text.apply(clean_pixels)
        .apply(clean_box_assign)
        .apply(remove_long_words)
        .apply(remove_urls)
    )

    return df


if __name__ == "__main__":
    df = load_corpus_as_dataframe(DOCUMENTS_DIR)
    df = df.sample(n=10)
    df = clean_text(df)
    breakpoint()

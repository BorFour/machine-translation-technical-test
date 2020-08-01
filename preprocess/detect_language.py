from typing import Union

import pandas as pd
from langdetect import detect

from .clean import DOCUMENTS_DIR, load_corpus_as_dataframe, clean_text


def _safely_detect_language(text: str) -> Union[str, None]:
    try:
        detected_language = detect(text)
    except Exception:  # FIXME: catch the actual exception instead of everything
        detected_language = None

    return detected_language


def detected_language(df: pd.DataFrame) -> pd.DataFrame:
    df["detected_language"] = df.cleaned_text.apply(_safely_detect_language)
    return df


if __name__ == "__main__":
    df = load_corpus_as_dataframe(DOCUMENTS_DIR)
    df = df.sample(n=10)
    df = clean_text(df)
    df = detected_language(df)
    breakpoint()

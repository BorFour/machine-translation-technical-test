from typing import Dict, List, Tuple
import os
import logging as log
from pprint import pprint

import tqdm
import pandas as pd


log.getLogger().setLevel(log.INFO)

DOCUMENTS_DIR = "./documents_challenge/"


def load_corpus_from_dir(corpus_dir: str) -> List[Tuple[str, str, str]]:
    dirs_and_filenames: List[Tuple[str, str]] = []

    for root, dirs, files in os.walk(DOCUMENTS_DIR, topdown=False):
        for name in files:
            dirs_and_filenames.append((root, name))

    corpus_dict = {}

    log.info(f"Loading corpus from {corpus_dir}")
    for root, name in tqdm.tqdm(dirs_and_filenames):
        filepath = os.path.join(root, name)
        with open(filepath) as f:
            corpus_dict[filepath] = f.read().strip()

    corpus_as_list = [(*k.split("/")[-3:], v) for k, v in corpus_dict.items()]

    return corpus_as_list


def load_corpus_as_dataframe(corpus_dir: str) -> pd.DataFrame:
    corpus = load_corpus_from_dir(corpus_dir)
    df = pd.DataFrame(corpus, columns=["context", "language", "docname", "text"])
    return df


def load_translation(context: str, language: str, docname: str) -> str:
    filepath = os.path.join(TRANSLATIONS_OUTPUT_DIR, context, language, docname)

    try:
        with open(filepath, "rt") as f:
            translation = f.read().strip()
    except FileNotFoundError:
        translation = None

    return translation


def load_translations_to_df(df: pd.DataFrame) -> pd.DataFrame:
    df["loaded_translation"] = df[["context", "language", "docname"]].apply(
        lambda x: load_translation(x.context, x.language, x.docname), axis=1
    )
    return df


if __name__ == "__main__":
    df = load_corpus_as_dataframe(DOCUMENTS_DIR)
    breakpoint()

from typing import Tuple, Union, List, Iterable
from collections import namedtuple

import pandas as pd

TranslationInput = Union[
    str, List[str], pd.Series
]  # The MarianMT model accept a single text or a list of texts
TranslationOutput = Union[str, List[str]]  # This is the same as the input

Document = str
TokenizedDocument = List[str]
Vocabulary = Iterable[str]

Prediction = namedtuple("Prediction", "class_ probability")

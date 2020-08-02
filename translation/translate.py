"""
SEE: https://huggingface.co/models?search=Helsinki-NLP

"""
from typing import Tuple, Union
from pprint import pprint

import pandas as pd
import torch
from transformers import MarianTokenizer, MarianMTModel

from custom_types import TranslationInput, TranslationOutput
from .available_languages import AVAILABLE_LANGUAGES


CACHED_MODELS = {}


def _load_marianmt(model_name: str) -> Tuple[MarianTokenizer, MarianMTModel]:
    """Load any MarianMT pretrained model and its tokenizer from its name."""

    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.cuda()

    return tokenizer, model


def _get_or_cache_model(language_code: str):
    global CACHED_MODELS

    model_name = AVAILABLE_LANGUAGES[language_code]
    cache = CACHED_MODELS.get("language_code")

    if not cache:
        tokenizer, model = _load_marianmt(model_name)
        CACHED_MODELS[language_code] = (tokenizer, model)
    else:
        tokenizer, model = cache

    return tokenizer, model


def _translate_text(
    text: TranslationInput, tokenizer: MarianTokenizer, model: MarianMTModel
) -> TranslationOutput:
    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        texts = text
    elif isinstance(text, pd.Series):
        texts = text.tolist()
    else:
        raise TypeError(f"Type {type(text)} not handled")

    batch = gen = None

    try:
        with torch.no_grad():
            batch = tokenizer.prepare_translation_batch(
                src_texts=texts, return_tensors="pt"
            )
            if torch.cuda.is_available():
                batch = batch.to("cuda")

            gen = model.generate(**batch)
            translations: List[str] = tokenizer.batch_decode(
                gen, skip_special_tokens=True
            )

        return translations
    finally:
        del batch
        del gen


def translate_text_to_spanish(
    text: TranslationInput, source_language: str
) -> TranslationOutput:
    tokenizer, model = _get_or_cache_model(source_language)
    return _translate_text(text, tokenizer, model)


def translate_english_text_to_spanish(text: TranslationInput) -> TranslationOutput:
    if isinstance(text, str):
        text = [text]

    tokenizer, model = _get_or_cache_model("en")
    return _translate_text([f">>es<< {t}" for t in text], tokenizer, model)


def translate_spanish_text_to_spanish(text: TranslationInput) -> TranslationOutput:
    # Let's just return the same text for now
    return text


def translate_to_spanish(
    text: TranslationInput, source_language: str
) -> TranslationOutput:
    if source_language == "es":
        return text
    elif source_language == "en":
        # English is a special case that needs to be handled independently
        return translate_english_text_to_spanish(text)
    else:
        if source_language in AVAILABLE_LANGUAGES.keys():
            return translate_text_to_spanish(text, source_language)
        else:
            raise ValueError(
                f"Translation from '{source_language}' to Spanish is not available"
            )


def translate_to_spanish_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df["translated_text"] = df.apply(
        lambda x: translate_to_spanish(x.cleaned_text, x.detected_language)[0], axis=1,
    )
    return df


if __name__ == "__main__":
    text = """Somebody once told me
The world is gonna roll me
I ain't the sharpest tool in the shed
She was looking kinda dumb
With her finger and her thumb
In shape of an "L" on her forehead."""

    translation = translate_to_spanish(text, "en")
    pprint(translation)

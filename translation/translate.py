"""
SEE: https://huggingface.co/models?search=Helsinki-NLP

"""
from typing import Tuple, Union
from pprint import pprint

import torch
from transformers import MarianTokenizer, MarianMTModel

from types import TranslationInput


LANGUAGE_MODEL_DICT = {
    "fr": "Helsinki-NLP/opus-mt-fr-es",
    "en": "Helsinki-NLP/opus-mt-en-ROMANCE",
}


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

    model_name = LANGUAGE_MODEL_DICT[language_code]
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


def translate_french_text_to_spanish(text: TranslationInput) -> TranslationOutput:
    tokenizer, model = _get_or_cache_model("fr")
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
        return translate_spanish_text_to_spanish(text)
    elif source_language == "fr":
        return translate_french_text_to_spanish(text)
    elif source_language == "en":
        return translate_english_text_to_spanish(text)
    else:
        raise ValueError(
            f"Translation from '{source_language}' to Spanish is not implemented"
        )


if __name__ == "__main__":
    text = """Somebody once told me
The world is gonna roll me
I ain't the sharpest tool in the shed
She was looking kinda dumb
With her finger and her thumb
In shape of an "L" on her forehead."""

    translation = translate_to_spanish(text, "en")
    pprint(translation)

import os


from multiprocessing.pool import ThreadPool
import tqdm

TRANSLATIONS_OUTPUT_DIR: str = "drive/My Drive/ea_test/translations_es/"


def save_translation(
    translation: str, context: str, origin_language: str, filename: str, output_dir: str
):
    output_dir: str = os.path.join(output_dir, context, origin_language)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, filename), "wt") as f:
        f.write(translation)


def make_translations(
    df: pd.DataFrame,
    source_language: str,
    batch_size: int = 15,
    offset: int = 0,
    limit: int = None,
):

    pool = ThreadPool(20)

    if limit is None:
        limit = len(df)

    for i in tqdm.tqdm(range(offset, limit, batch_size)):

        try:
            # print(f"Slicing with {i}:{i+batch_size}")
            batch_df = df[df.cleaned_text.str.len() > 0].iloc[i : i + batch_size]
            # print(f"Batch length: {len(batch_df)}")

            if len(batch_df) != batch_size:
                print(
                    f"WARNING: batch length {len(batch_df)} is not equal to batch size {batch_size} "
                    f"at slice {i}:{i+batch_size}"
                )

            translation = translate_to_spanish(
                batch_df.cleaned_text.tolist(), source_language=source_language
            )

            POOL.imap_unordered(
                lambda x: save_translation(x[0], x[1], x[2], x[3]),
                zip(translation, batch_df.context, batch_df.language, batch_df.docname),
            )
        except Exception:
            print(f"Failed at batch {i}")
            raise


def asdfasd():
    french_df = df[df.language == "fr"]
    english_df = df[df.language == "en"]
    spanish_df = df[df.language == "es"]

    # French to Spanish translations
    # make_translations(french_df, source_language="fr", batch_size=15, offset=2460, limit=int(len(french_df) / 2))
    # make_translations(french_df, source_language="fr", batch_size=15, offset=int(len(french_df) / 2))

    # English to Spanish translations
    make_translations(
        english_df,
        source_language="en",
        batch_size=15,
        offset=0,
        limit=int(len(english_df) / 2),
    )
    # make_translations(english_df, source_language="en", batch_size=15, offset=int(len(english_df) / 2))

    # Spanish to Spanish translations (do nothing but save them to disk)
    # make_translations(spanish_df, source_language="es", batch_size=15)

"""
Generate a backtranslated version of the data
"""

import argparse
from BackTranslation import BackTranslation
from data import load_full_train
import pandas as pd
from spacy.lang.en import English
from time import sleep
from tqdm import tqdm


def backtranslate_dataset(output_file, source_lang="en", target_lang="es"):
    """Backtranslates the full training dataset to generate a new one"""
    print(f"Backtranslating from {source_lang} to {target_lang}")
    full = load_full_train()
    backtranslated = backtranslate_texts(full["description"], source_lang, target_lang)
    backtranslated_df = pd.DataFrame(
        index=full.index,
        data={
            "name": full["name"],
            "description": backtranslated
        }
    )
    backtranslated_df.to_csv(output_file, index=False)


def backtranslate_texts(texts, source_lang="en", target_lang="es"):
    """Backtranslates an iterable of texts between two languages"""
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    backtranslator = BackTranslation()
    backtranslations = []
    for text in tqdm(texts):
        # Split text into sentences
        sentences = [sent.string.strip() for sent in nlp(text).sents]
        # Back translate each sentence
        sentences_backtranslations = []
        for sentence in sentences:
            translated = False
            while not translated:
                try:
                    backtranslation = backtranslator.translate(sentence, src=source_lang, tmp=target_lang)
                    sentences_backtranslations.append(backtranslation.result_text.lower())
                    translated = True
                except:
                    sleep(1)
        # Join backtranslations
        backtranslations.append(" ".join(sentences_backtranslations))
    return backtranslations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates backtranslated samples for the full training dataset')
    parser.add_argument('output_file', type=str, help='output file in which to write backtranslated texts')
    parser.add_argument('--source_lang', type=str, default="en", help='source language')
    parser.add_argument('--target_lang', type=str, default="es", help='target language')
    args = parser.parse_args()
    backtranslate_dataset(args.output_file, args.source_lang, args.target_lang)

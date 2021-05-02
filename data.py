"""Data preparacion functions"""

import json
import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit


def load_data(random_state=42, sample=None, backtranslation=None):
    """Load training, validation and test data folds

    Arguments:
        random_state: random seed for train/validation split
        sample: maximum number of data points to subsample for train and validation data
        backtranslation: list of language codes to load additional backtranslated samples.

    Returns training, validation and test dataframes.
    """
    # Load original training data
    train = pd.read_csv("train.csv")
    train = train.apply(normalize_texts, axis=0)

    # Validation split respecting groups
    gss = GroupShuffleSplit(n_splits=2, test_size=0.1, random_state=random_state)
    train_idx, val_idx = next(gss.split(train, groups=train["name"]))
    val = train.iloc[val_idx]
    train = train.iloc[train_idx]

    # Load crawls, add them to training data
    crawl_files = [
        "crawleddata.json",  # v2
        "crawleddata_home.json",  # v3
        "crawleddata_extra.json",  # v4 
        "crawleddata_multiweb.json",  # v5
        "crawleddata_yahoo.json",  # v5
        "crawleddata_multiweb_v2.json"  # v6
    ]
    for crawl_file in crawl_files:
        crawled = load_crawl_file(crawl_file)
        train = train.append(crawled)
    train = train.drop_duplicates()

    # Add backtranslated samples for training data, if requested
    if backtranslation is not None:
        for lang in backtranslation:
            backtranslated_data = pd.read_csv(f"backtranslation_{lang}.csv")
            train = train.append(backtranslated_data.iloc[train_idx])

    # Reshuffle training data to mix crawled and backtranslated samples with original samples
    train = train.sample(frac=1., random_state=123)

    # Remove from train those elements that appear in val
    mask = train["name"].isin(val["name"]) | train["description"].isin(val["description"])
    train = train[~mask]

    # Subsample if requested
    if sample is not None:
        train = train[:sample]
        val = val[:sample]

    test = pd.read_csv("test_descriptions.csv")
    test = test.apply(normalize_texts, axis=0)

    return train, val, test


def load_full_train():
    train = pd.read_csv("train.csv")
    train = train.apply(normalize_texts, axis=0)

    #crawl_files = ["crawleddata.json", "crawleddata_home.json", "crawleddata_extra.json"]
    crawl_files = ["crawleddata.json", "crawleddata_home.json", "crawleddata_extra.json", "crawleddata_multiweb.json", "crawleddata_yahoo.json"]
    for crawl_file in crawl_files:
        crawled = load_crawl_file(crawl_file)
        train = train.append(crawled)

    return train.drop_duplicates().sample(frac=1., random_state=123)


def load_crawl_file(filepath):
    """Loads a crawl file from disk, removing bad elements"""
    # Load file
    with open(filepath) as f:
        crawled = json.load(f)
    crawled = pd.DataFrame(crawled)
    crawled = crawled.apply(normalize_texts, axis=0)
    # Remove bad items
    crawled = crawled[crawled["description"] != "this item has "]
    return crawled


def load_train_val_idx(random_state=42):
    train = load_full_train()

    # Split into train and validation, making sure no product names are shared between the two of them
    gss = GroupShuffleSplit(n_splits=2, test_size=0.1, random_state=random_state)
    train_idx, val_idx = next(gss.split(train, groups=train["name"]))

    return train_idx, val_idx


def normalize_texts(texts):
    return [normalize_text(text) for text in texts]


def normalize_text(text):
    # To lower case
    text = text.lower()
    # Remove html tags
    text = text.replace("<br/>", "")
    # Remove useless sections at the end of texts
    to_remove = ["height of model", "model height", "join life care for water"]
    for substring in to_remove:
        if substring in text:
            text = text[:text.find(substring)]
    # Zara Home crawl corrections
    corrections = [
        ("\u00e9", "é"),
        ("\u00a0", " "),
        ("\u00ae", "®"),
        ("\u00c9", "É"),
        ("\u00ba", "º"),
        ("\u2019", "’"),
        ("\u00b2", "²"),
        ("\u2022", "\"")
    ]
    for orig, target in corrections:
        text = text.replace(orig, target)
    return text

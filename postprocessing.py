"""Functions for post-processing model results"""
from itertools import chain, islice
import os
import pandas as pd
import shutil
from tempfile import TemporaryDirectory
import torch


from model import splitevery


def names_from_train(descriptions, train):
    """Generate product names for test descriptions that are exactly the same as training descriptions"""
    name_proposals = []
    unique_descriptions = set(train["description"])
    for description in descriptions:
        if description in unique_descriptions:
            name_proposals.append(list(train[train["description"] == description]["name"].values))
        else:
            name_proposals.append([])
    return name_proposals


def merge_name_proposals(proposals_lists, top=10):
    """Merges an iterable of name proposals. Names in the first iterables are given priority"""
    merged = [list(chain(*proposals)) for proposals in zip(*proposals_lists)]
    # Remove duplicates
    dedup = []
    for names in merged:
        proposals = []
        for elem in names:
            if elem not in proposals:
                proposals.append(elem)
        dedup.append(proposals[:top])
    return dedup


def save_submission(proposed_names, file_name, zip=True):
    """Saves predictions in the submission file format
    
    Arguments:
        proposed_names: list of name proposals, one per article. Each element must be a list of strings.
        file_name: name of the file in which to save the predictions, without extension.
        zip: whether to save submission in file format. If false, save it in raw CSV.
    """
    tmpdir = TemporaryDirectory()
    csv_name = tmpdir.name + "/submission.csv"
    with open(csv_name, "w") as f:
        f.write("name\n")
        for name_list in proposed_names:
            f.write(",".join(name_list) + "\n")
    if zip:
        shutil.make_archive(f'{file_name}.csv', 'zip', tmpdir.name)
    else:
        os.rename(csv_name, f'{file_name}.csv')


def read_proposals_files(proposals_files, remove_duplicates="local"):
    """Reads a number of proposal files and joins their proposals"""
    assert remove_duplicates in ("local", "global", None)
    name_candidates = []
    for pfile in proposals_files:
        with open(pfile) as f:
            name_candidates.append([line[:-1].split(",") for line in f.readlines()][1:])
    # Join name lists
    if remove_duplicates == "global":
        return [list(set.union(*[set(e) for e in elems])) for elems in zip(*name_candidates)]
    elif remove_duplicates == "local":
        return [list(chain(*[list(set(e)) for e in elems])) for elems in zip(*name_candidates)]
    else:
        return [list(chain(*elems)) for elems in zip(*name_candidates)]


def normalize_proposals(proposals):
    """Performs a number of normalizations in name proposals"""
    # Remove trailing and leading whitespaces
    proposals = [p.strip() for p in proposals]
    # Discard proposals that start with a single character not in a recognized list
    VALID_STARTERS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'v', '©', '®', '™'}
    proposals = [p for p in proposals if len(p.split(" ")[0]) > 1 or p.split(" ")[0] in VALID_STARTERS]
    # Remove duplicates, preserving order
    proposals = [p for i, p in enumerate(proposals) if proposals[:i].count(p) == 0]

    return proposals


def normalize_proposals_list(proposals_list):
    """Performs a number of normalizations on a list of name proposals"""
    return [normalize_proposals(proposals) for proposals in proposals_list]


def unroll_data(names, descriptions, name_proposals):
    """Transforms a dataset of name+description+name_proposals into a dataset of name+description+label
    
    Positive labels are those real name+description pairs, negative labels are wrongly proposed names paired with descriptions.
    """
    new_names, new_descriptions, labels = [], [], []
    for name, description, proposals in zip(names, descriptions, name_proposals):
        # Positive label
        new_names.append(name)
        new_descriptions.append(description)
        labels.append(1)
        # Negative labels
        proposals = list(set(proposals) - {name})
        n_negative = len(proposals)
        new_names.extend(proposals)
        new_descriptions.extend([description] * n_negative)
        labels.extend([0] * n_negative)
    return pd.DataFrame({
        "name": new_names,
        "description": new_descriptions,
        "label": labels
    })


def unroll_test_data(descriptions, name_proposals):
    """Similar unroll_data but for test data. Labels are not generated, just all crossings of descriptions and name proposals"""
    new_names, new_descriptions, original_rows = [], [], []
    for i, elems in enumerate(zip(descriptions, name_proposals)):
        description, proposals = elems
        n_negative = len(proposals)
        new_names.extend(proposals)
        new_descriptions.extend([description] * n_negative)
        original_rows.extend([i] * n_negative)
    return pd.DataFrame({
        "name": new_names,
        "description": new_descriptions,
        "original_row": original_rows
    })


def reroll_score_test_data(unrolled_test, ascending=False):
    """DO A BARREL ROLL!!!!"""
    srt = unrolled_test.groupby('original_row', group_keys=False).apply(lambda x: x.sort_values(by="score", ascending=ascending).head(10))
    proposals = srt.groupby("original_row")["name"].apply(lambda x: list(x)[:10]).reset_index(name='proposals')["proposals"]
    return proposals


def model_logprobs(model, collator, input_texts, output_texts, length_penalty=0, batchsize=128):
    """Computes logprobabilities of input and output texts for a given model

    Inputs:
        model: conditional generation model to use
        collator: collator to process texts
        input_texts: iterator of input texts to evaluate
        output_texts: iterator of output texts to evaluate
        length_penalty: penalty to apply to shorter texts, see https://www.aclweb.org/anthology/W18-6322.pdf
        batchsize: size of batch to use for grouping texts when evaluating the model
    """
    total_logprobs = []
    for batch in splitevery(zip(input_texts, output_texts), batchsize):
        input_batch, output_batch = zip(*batch)
        encoded_inputs = collator.encode_inputs(input_batch)
        encoded_outputs = collator.encode_outputs(output_batch)
        with torch.no_grad():
            output = model(**encoded_inputs, **encoded_outputs)
        # Normalize probabilities
        normalized = output["logits"].log_softmax(dim=2)
        for k in range(len(input_batch)):
            total_logprob = sum([
                normalized[k, i, encoded_outputs["labels"][k][i]]
                for i in range(1, len(encoded_outputs["labels"][k]) - 1)
            ]).cpu().numpy().item()
            # Normalize by length: https://www.aclweb.org/anthology/W18-6322.pdf
            num_output_tokens = len(encoded_outputs["labels"][k]) - 2  # Ignore tokens for text start/end
            total_logprob /= (5+num_output_tokens)**length_penalty / (5+1)**length_penalty
            total_logprobs.append(total_logprob) 
    return total_logprobs



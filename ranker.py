"""Functions to build and use a ranking model"""

from collections import defaultdict, Counter
from itertools import chain
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback


from model import CoprocessorMixin
from postprocessing import unroll_data


class SpainAIRankerCollator(CoprocessorMixin):
    """Specialized version of the data collator for ranking"""

    def __init__(self, tokenizer):
        """Initializes the collator with a tokenizer"""
        self.tokenizer = tokenizer
        super().__init__()

    def encode_inputs(self, texts1, texts2):
        """Transforms two iterables of input texts into a dictionary of model input tensors, stored in the GPU
        
        Texts in iterators are zipped together using a [SEP] token between them.
        """
        joined = [f"{t1} {self.tokenizer.sep_token} {t2}" for t1, t2 in zip(texts1, texts2)]
        input_encodings = self.tokenizer.batch_encode_plus(joined, padding="longest", truncation=True, return_tensors="pt")
        return self.to_device({
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        })

    def encode_outputs(self, targets):
        return self.to_device({
            'labels': torch.tensor(targets).long(),
        })

    def __call__(self, patterns):
        """Collate a batch of patterns
        
        Arguments:
            - patterns: iterable of tuples in the form (text1, text2, targets),
                or just iterable of tuples in the form (text1, text2)
            
        Output: dictionary of torch tensors ready for model input
        """
        # Check kind of input
        if len(patterns) < 1: raise ValueError(f"At least one pattern is required, found {len(patterns)}")
        if not isinstance(patterns[0], tuple): raise ValueError(f"Each pattern must be a tuple with two texts, or a tuple with two texts and a class. Found {patterns[0]}")
        targets_provided = len(patterns[0]) == 3
        # Split texts and targets from the input list of tuples
        if targets_provided:
            texts1, texts2, targets = zip(*patterns)
        else:
            texts1, texts2 = zip(*patterns)
        # Encode inputs
        tensors = self.encode_inputs(texts1, texts2)
        # Encode outputs (if provided)
        if targets_provided:
          tensors = {**tensors, **self.encode_outputs(targets)}
        return tensors


class SpainAIRankerDataset(Dataset):
    """Dataset for learning a ranker of name+description pairs
    
    The dataset is built from three types of patterns:
        - Positive patterns: real alignments of name+description found in the data
        - Negative proposals: pairs of name+description where the name was wrongly proposed by another algorithm
        - Negative samples: pairs of name+description where the name is a sample at random from other iten in the data
    """

    def __init__(self, descriptions, names, name_proposals=None, negative_sampling_factor=1, training_mode=True):
        assert len(descriptions) == len(names), "Descriptions and names must contain the same number of items"

        self.descriptions = descriptions
        self.names = names
        self.negative_sampling_factor = negative_sampling_factor
        self.training_mode = training_mode

        # Prepare negative proposals: remove all that match with the real name
        if name_proposals is not None:
            assert len(descriptions) == len(name_proposals), "Descriptions and names proposals must contain the same number of items"
            self.name_proposals = [
                [proposal for proposal in proposals if proposal != name]
                for proposals, name in zip(name_proposals, names)
            ]
        else:
            self.name_proposals = [[] for _ in range(len(descriptions))]

        # Prepare indices for locating each generated pattern
        self.pattern_boundaries = list(np.cumsum([0] + [1 + len(proposals) + self.negative_sampling_factor for proposals in self.name_proposals]))


    def __len__(self):
        if self.training_mode:
            return self.pattern_boundaries[-1]
        else:
            return len(self.names)

    def __getitem__(self, idx):
        """Returns the idx-th pattern in the dataset

        For each original pattern, the data is sorted as:
            - First, the positive pattern
            - Next, the negative proposals
            - Next, the negative samples
        """
        if not self.training_mode:
            return self.descriptions[idx], self.names[idx]

        # Find original pattern relating to this idx
        original_idx = list(np.array(self.pattern_boundaries) > idx).index(True) - 1
        description = self.descriptions[original_idx]
        generator_start = self.pattern_boundaries[original_idx]
        generator_offset = idx - generator_start

        # Positive pattern
        if generator_offset == 0:
            name = self.names[original_idx]
            label = 1
        # Negative proposals
        elif generator_offset <= len(self.name_proposals[original_idx]):
            name = self.name_proposals[original_idx][generator_offset-1]
            label = 0
        # Negative samples
        else:
            name = np.random.choice(self.names)
            label = 0

        return description, name, label


class AccuracyCallback(TrainerCallback):
    """Callback that measures the accuracy over the validation set"""
    def on_evaluate(self, args, state, control, model, metrics, eval_dataloader, **kwargs):
        hits = 0
        n = 0
        for batch in eval_dataloader:
            with torch.no_grad():
                preds = model(**batch)
            hits += sum(preds.logits.argmax(dim=1) == batch["labels"]).cpu().numpy()
            n += len(batch["labels"])
        accuracy = hits / n
        metrics["eval_accuracy"] = accuracy
        print(f"accuracy={accuracy}")


class AUCCallback(TrainerCallback):
    """Callback that measures Area Under the (ROC) Curve over the validation set"""
    def on_evaluate(self, args, state, control, model, metrics, eval_dataloader, **kwargs):
        labels = []
        probs = []
        for batch in eval_dataloader:
            with torch.no_grad():
                preds = model(**batch)
            probs.extend(list(preds.logits[:, 1].cpu().numpy()))
            labels.extend(list(batch["labels"].cpu().numpy()))
        auc = roc_auc_score(labels, probs)
        metrics["eval_auc"] = auc
        print(f"AUC={auc}")


def train_ranker(model_name, train, val, train_batch_size=32, gradient_accumulation_steps=1):
    # Unroll datasets
    unrolled_train = unroll_data(train["name"], train["description"], train["name_proposals"])
    unrolled_val = unroll_data(val["name"], val["description"], val["name_proposals"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    collator = SpainAIRankerCollator(tokenizer)
    accuracy_callback = AccuracyCallback()

    training_args = TrainingArguments(
        output_dir='./ranker',
        #num_train_epochs=epochs,
        max_steps=100000,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=32,
        warmup_steps=500,               
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=1000,
        load_best_model_at_end=True,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,        
        data_collator=collator,               
        args=training_args,                  
        train_dataset=list(zip(unrolled_train['name'].values, unrolled_train['description'].values, unrolled_train['label'].values)),
        eval_dataset=list(zip(unrolled_val['name'].values, unrolled_val['description'].values, unrolled_val['label'].values)),
        #callbacks=[accuracy_callback]  # TODO: replace accuracy with AUC? might be better for unbalanced dataset
        callbacks=[AUCCallback]
    )

    trainer.train()

    return trainer, model, tokenizer, collator


def load_ranker(model_path, model_class):
    """Loads a previously trained Ranker model from file
    
    Arguments:
        model_path: path to the model checkpoint to load
        model_class: model class used for training
    """
    tokenizer = AutoTokenizer.from_pretrained(model_class)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    collator = SpainAIRankerCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir='./ranker',
        per_device_eval_batch_size=32,
        #disable_tqdm=True
    )

    trainer = Trainer(
        model=model,        
        data_collator=collator,               
        args=training_args,
        train_dataset=SpainAIRankerDataset(["dummy"], ["dummy"])
    )

    return trainer


def rank_candidates(ranker, descriptions, name_proposals, max_candidates=10):
    """Given lists of product descriptions and corresponding name proposals, sorts the proposals using a ranker"""
    ranked_proposals = []

    for description, proposals in zip(descriptions, name_proposals):
        test = SpainAIRankerDataset([description] * len(proposals), proposals, training_mode=False)
        scores = ranker.predict(test).predictions[:, 1]
        sorted_scores = sorted(zip(proposals, scores), key=lambda couple: couple[1], reverse=True)
        sorted_candidates = [x[0] for x in sorted_scores]
        ranked_proposals.append(sorted_candidates[:max_candidates])

    return ranked_proposals


def majority_ranking(proposals):
    """Ranks a list of proposals by using majority voting

    Arguments:
        proposals: iterable of proposals candidates
    """
    return [[elem[0] for elem in Counter(names).most_common(10)] for names in proposals]


def add_dcg_weights(iterable):
    """Transforms an iterable into an iterable of tuples (DGC_weight, original_element)"""
    return [(1. / np.log2(i+2), x) for i, x in enumerate(iterable)]


def _sort_by_weights(weight_iterable):
    """Sorts in decreasing order an iterable of tuples in the form (weight, element).

    Returns a list of elements, already sorted. If an elements appears more than once, its weights are summed up
    """
    total_weights = defaultdict(int)
    for weigth, key in weight_iterable:
        total_weights[key] += weigth
    return sorted(total_weights, key=lambda k: total_weights[k], reverse=True)


def dcg_ranking(proposals, max_candidates=10):
    """Ranks a list of lists proposals by weighing them with Discounted Cumulative Gain weights

    Arguments:
        proposals: iterable of proposals candidates lists
    """
    return [
        _sort_by_weights(chain(*[add_dcg_weights(proposals_list) for proposals_list in proposals_lists]))[:max_candidates]
        for proposals_lists in proposals
    ]

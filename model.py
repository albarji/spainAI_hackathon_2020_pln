"""Model functions"""

from itertools import islice
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration, PegasusForConditionalGeneration, ProphetNetForConditionalGeneration
from transformers import TrainerCallback, Trainer, TrainingArguments, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right


class CoprocessorMixin():
    """Mixin that provides functionality to move Pytorch tensors to a coprocessor"""

    def __init__(self):
        """Initializes the coprocessor, looking for a GPU if available"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_device(self, tensors):
        """Moves to the computing device a dictionary of Pytorch tensors"""
        for key in tensors:
            tensors[key] = tensors[key].to(self.device)
        return tensors


class SpainAICollator(CoprocessorMixin):
    """Data collator for SpainAI hackathon data"""
    
    def __init__(self, tokenizer, model):
        """Initializes the collator with a tokenizer"""
        self.tokenizer = tokenizer
        self.model = model
        super().__init__()
    
    def encode_inputs(self, texts):
        """Transforms an iterable of input texts into a dictionary of model input tensors, stored in the GPU"""
        input_encodings = self.tokenizer.batch_encode_plus(list(texts) if isinstance(texts, tuple) else texts, padding="longest", truncation=True, return_tensors="pt")
        return self.to_device({
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        })

    def encode_outputs(self, texts):
        target_encodings = self.tokenizer.batch_encode_plus(list(texts) if isinstance(texts, tuple) else texts, padding="longest", truncation=True, return_tensors="pt")
        labels = target_encodings['input_ids']
        decoder_input_ids = shift_tokens_right(labels, self.model.config.pad_token_id, self.model.config.decoder_start_token_id)
        labels[labels[:, :] == self.model.config.pad_token_id] = -100
        return self.to_device({
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        })
    
    def __call__(self, patterns):
        """Collate a batch of patterns
        
        Arguments:
            - patterns: iterable of tuples in the form (input_text, output_text), 
              or just iterable of input texts
            
        Output: dictionary of torch tensors ready for model input
        """
        # Check kind of input
        if len(patterns) < 1: raise ValueError(f"At least one pattern is required, found {len(patterns)}")
        if not isinstance(patterns[0], (tuple, str)): raise ValueError(f"Each pattern must be one text, or a tuple with two texts. Found {patterns[0]}")
        targets_provided = len(patterns[0]) == 2
        # Split texts and classes from the input list of tuples
        if targets_provided:
            input_texts, output_texts = zip(*patterns)
        else:
            input_texts = patterns
        # Encode inputs
        tensors = self.encode_inputs(input_texts)
        # Encode outputs (if provided)
        if targets_provided:
          tensors = {**tensors, **self.encode_outputs(output_texts)}
        return tensors


def splitevery(iterable, n):
    """Returns blocks of elements from an iterator"""
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def generate_names(model, tokenizer, collator, descriptions, num_sequences=30, batchsize=4, deduplicate=True, max_candidates=10):
    name_proposals = []
    # Generate predictions in batches
    for descriptions in splitevery(descriptions, batchsize):
        tensors = collator.encode_inputs(descriptions)
        summary_ids = model.generate(
            tensors['input_ids'], 
            num_beams=num_sequences, 
            num_return_sequences=num_sequences, 
            early_stopping=True,
            #top_k=50, 
            #top_p=0.95,
            #length_penalty=0,
        )
        decoded = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        # Group proposals for each element
        decoded = splitevery(decoded, num_sequences)
        # Remove duplicates
        filtered = []
        for line in decoded:
            filtered_line = []
            if deduplicate:
                for elem in line:
                    if elem not in filtered_line:
                        filtered_line.append(elem)
            else:
                filtered_line = line
            filtered.append(filtered_line[:max_candidates])
        name_proposals.extend(filtered)
    return name_proposals


def dcg(proposals, targets):
    """Computes Discounted Cumulative Gain for a list of proposals and expected targets"""
    score = 0
    for proposal, target in zip(proposals, targets):
        if target in proposal:
            idx = proposal.index(target)
            score += 1 / np.log2(idx+2) 
    return score / len(proposals) * 100


class DCGCallback(TrainerCallback):
    def __init__(self, tokenizer, collator, val):
        self.tokenizer = tokenizer
        self.collator = collator
        self.val = val

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        # Generate names
        names = generate_names(model, self.tokenizer, self.collator, self.val["description"])
        proposals_val = pd.DataFrame({
            "description": self.val["description"],
            "name": self.val["name"],
            "proposed": names,
        })
        # Evaluate DGC for each name  # TODO: use function above
        dcg = 0
        for _, row in proposals_val.iterrows():
            if row["name"] in row["proposed"]:
                idx = row["proposed"].index(row["name"])
                dcg += 1 / np.log2(idx+2)
        dcg = dcg / len(proposals_val) * 100
        metrics["eval_dcg"] = dcg
        print(f"DCG={dcg}")


def load_model(model_name, pre_trained_class=None):
    model_class = model_name if not pre_trained_class else pre_trained_class 
    if model_class.startswith("facebook/bart"):
        return BartForConditionalGeneration.from_pretrained(model_name)
    elif model_class.startswith("google/pegasus"):
        return PegasusForConditionalGeneration.from_pretrained(model_name)
    elif model_class.startswith("t5"):
        return T5ForConditionalGeneration.from_pretrained(model_name)
    elif model_class.startswith("microsoft/prophetnet"):
        return ProphetNetForConditionalGeneration.from_pretrained(model_name)
    else:
        raise ValueError(f"Unrecognized model family for model {model_name}")


def train_model(model_name, train, val, train_batch_size=32, gradient_accumulation_steps=1, epochs=1, pre_trained_class=None, warmup_steps=500, weight_decay=0.01):
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_class if pre_trained_class else model_name)
    model = load_model(model_name, pre_trained_class)
    collator = SpainAICollator(tokenizer, model)
    dgc_callback = DCGCallback(tokenizer, collator, val)

    training_args = TrainingArguments(
        output_dir='./model',
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=32,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,        
        data_collator=collator,               
        args=training_args,                  
        train_dataset=list(zip(train['description'], train['name'])),
        eval_dataset=list(zip(val['description'], val['name'])),
        #callbacks=[dgc_callback]
    )

    if pre_trained_class:
        trainer.train(model_name)
    else:
        trainer.train()

    return trainer, model, tokenizer, collator


def train_and_score_model(model_name, train, val, **kwargs):
    """Trains a model with the given parameters and returns DCG score over the validation data"""
    trainer, model, tokenizer, collator = train_model(model_name, train, val, **kwargs)
    names = generate_names(model, tokenizer, collator, val["description"])
    score = dcg(names, val["name"])

    del trainer
    del model
    del tokenizer
    del collator
    torch.cuda.empty_cache()

    return score


from functools import partial
from skopt import gp_minimize

def _unwrap_model_parameters(params):
    return {
        "warmup_steps": int(params[0]), 
        "weight_decay": float(params[1])
        ## TODO: add attention and hidden dropouts
    }

def _wrap_model_parameters(warmup_steps, weight_decay):
    return [
        warmup_steps,
        weight_decay
    ]

def optimize_model_parameters(model_name, train, val, train_batch_size=32, epochs=1, gradient_accumulation_steps=1, n_calls=15):
    # Prepare scoring function
    partial_scorer = partial(
        train_and_score_model, 
        model_name=model_name,
        train=train, 
        val=val,
        train_batch_size=train_batch_size,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    scorer = lambda params: partial_scorer(**_unwrap_model_parameters(params))

    results = gp_minimize(
        scorer,
        dimensions=[
            [100, 200, 500, 1000, 2000],  # Warmup steps
            [0, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]  # Weight decay
        ],
        n_calls=n_calls,
        verbose=True
    )

    print("Tested parameters and scores: ", zip(results.x_iters, results.func_vals))
    print("Best parameters and score: ", results.x, results.fun)

    return _unwrap_model_parameters(results.x)


#### TRYING TO CREATE SKLEARN CLASS HERE

class TransformersConditionalGenerator(BaseEstimator, ClassifierMixin):
    """Class implementing a Transformers-based model for conditional generation

    Follow sklearn standard to allow for easy hyperparameter tuning.
    """
    def __init__(self, model_name, learning_rate=5e-5, epochs=1, train_batch_size=8, attention_dropout_prob=0.1, 
        hidden_dropout_prob=0.1, warmup_steps=500, weight_decay=0.01, gradient_accumulation_steps=1, pre_trained_class=None):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.pre_trained_class = pre_trained_class

    def _load_base_config(self):
        model_class = self.model_name if not self.pre_trained_class else self.pre_trained_class 
        if model_class.startswith("facebook/bart"):
            return BartConfig.from_pretrained(self.model_name)
        elif model_class.startswith("google/pegasus"):
            return PegasusConfigu.from_pretrained(self.model_name)
        elif model_class.startswith("t5"):
            return T5Configu.from_pretrained(self.model_name)
        elif model_class.startswith("microsoft/prophetnet"):
            return ProphetNetConfig.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unrecognized model family for model {model_class}")

    def _load_model(self, config):
        model_class = self.model_name if not self.pre_trained_class else self.pre_trained_class 
        if model_class.startswith("facebook/bart"):
            return BartForConditionalGeneration.from_pretrained(self.model_name, config=config)
        elif model_class.startswith("google/pegasus"):
            return PegasusForConditionalGeneration.from_pretrained(self.model_name, config=config)
        elif model_class.startswith("t5"):
            return T5ForConditionalGeneration.from_pretrained(self.model_name, config=config)
        elif model_class.startswith("microsoft/prophetnet"):
            return ProphetNetForConditionalGeneration.from_pretrained(self.model_name, config=config)
        else:
            raise ValueError(f"Unrecognized model family for model {model_class}")

    def fit(self, train, val):
        # Clear GPU memory
        torch.cuda.empty_cache()
        # Prepare config
        config = self._load_base_config()
        config.attention_dropout_prob = self.attention_dropout_prob
        config.hidden_dropout_prob = self.hidden_dropout_prob
        print(config)  # FIXME: check correct naming of probabilities
        # Prepare tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.pre_trained_class if self.pre_trained_class else self.model_name)
        # Prepare model
        model = self.load_model(config)
        # Prepare collator
        collator = SpainAICollator(tokenizer, model)
        # Prepare DCG callback
        dcg_callback = DCGCallback(tokenizer, collator, val)

        training_args = TrainingArguments(
            output_dir='./model',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            per_device_eval_batch_size=32,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            evaluation_strategy="epoch",
            metric_for_best_model="eval_dcg",
            greater_is_better=True,
            load_best_model_at_end=True,
            disable_tqdm=True
        )

        trainer = Trainer(
            model=model,        
            data_collator=collator,               
            args=training_args,                  
            train_dataset=list(zip(train['description'], train['name'])),
            eval_dataset=list(zip(val['description'], val['name'])),
            callbacks=[dcg_callback]
        )

        if pre_trained_class:
            trainer.train(model_name)
        else:
            trainer.train()

        return trainer, model, tokenizer, collator

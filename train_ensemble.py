"""Train an ensemble of model of the same class"""

import numpy as np
from numpy.random import default_rng
import pandas as pd
import torch
import transformers


from data import load_data
from model import generate_names, train_model
from postprocessing import save_submission


# Config
transformers.logging.set_verbosity_info()


rg = default_rng(12345)

# Iterate over ensemble models
n_models = 1000
for i in range(n_models):
    # Prepare data
    seed = rg.integers(0, 2**32 - 1, endpoint=False)
    # Skip already generated models
    if i <= 100:
        continue
    train, val, test = load_data(random_state=seed)
    # Train model
    #trainer, model, tokenizer, collator = train_model('facebook/bart-base', train, val, train_batch_size=16, gradient_accumulation_steps=4, epochs=20)  # Generate batch 16
    #trainer, model, tokenizer, collator = train_model('t5-base', train, val, train_batch_size=8, gradient_accumulation_steps=8, epochs=25)
    #trainer, model, tokenizer, collator = train_model('t5-small', train, val, train_batch_size=8, gradient_accumulation_steps=8, epochs=60)
    trainer, model, tokenizer, collator = train_model('facebook/bart-large', train, val, train_batch_size=4, gradient_accumulation_steps=16, epochs=5)  # Generate batch 4
    #trainer, model, tokenizer, collator = train_model('google/pegasus-large', train, val, train_batch_size=2, gradient_accumulation_steps=32, epochs=20)
    # Save model
    trainer.save_model(f"./model_ensemble/{i}")
    # Generate predictions
    names = generate_names(model, tokenizer, collator, test["description"])
    save_submission(names, f"./model_ensemble/submission_{i}", zip=False)
    del trainer
    del model
    del tokenizer
    del collator
    torch.cuda.empty_cache()


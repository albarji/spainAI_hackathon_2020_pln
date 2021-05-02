"""Train model"""

import pandas as pd
import transformers


from data import load_data
from model import generate_names, train_model
from postprocessing import save_submission


# Config
transformers.logging.set_verbosity_info()

# Prepare data
#train, val, test = load_data(backtranslation=["es"])
train, val, test = load_data()

# Train model
#trainer, model, tokenizer, collator = train_model('microsoft/prophetnet-large-uncased', train[:64], val, train_batch_size=4, gradient_accumulation_steps=16, epochs=30)
#trainer, model, tokenizer, collator = train_model('t5-large', train, val, train_batch_size=1, gradient_accumulation_steps=64, epochs=30)
#trainer, model, tokenizer, collator = train_model('./model/checkpoint-31380/', train, val, train_batch_size=4, gradient_accumulation_steps=8, epochs=60, pre_trained_class='t5-small')
trainer, model, tokenizer, collator = train_model('facebook/bart-base', train, val, train_batch_size=16, gradient_accumulation_steps=2, epochs=20)

# Generate predictions
names = generate_names(model, tokenizer, collator, test["description"])
save_submission(names, "submission_xx")
save_submission(names, "submission_xx", zip=False)

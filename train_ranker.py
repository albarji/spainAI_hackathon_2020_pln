import transformers

from data import load_data
from postprocessing import read_proposals_files
from ranker import train_ranker


# Config
transformers.logging.set_verbosity_info()

# Load data
train, val, test = load_data()

files = [
    "BART-base-submission-23.csv", 
    "BART-large-submission-21.csv",
    "pegasus-large-submission-30.csv",
    "t5-base-submission-40.csv",
    "t5-small-submission-36.csv",
]

train["name_proposals"] = read_proposals_files([f"train_{file}" for file in files])
val["name_proposals"] = read_proposals_files([f"val_{file}" for file in files])

# Train ranker
trainer, model, tokenizer, collator = train_ranker("distilbert-base-uncased", train, val, train_batch_size=32, gradient_accumulation_steps=2)

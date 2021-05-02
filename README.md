# Spain AI 2020 NLP Hackathon

This repository contains the code and notebooks I used to win the second prize at Natural Language Processing task of the Spain AI 2020 hackathon. All the materials are published in their raw form, exactly as I used them in the competition, so expect a high degree of chaos, cryptic notes to myself, and low-quality code.

Some general navigation guidelines:

* **environment.yml**: Anaconda environment with all packages used.
* **data.py**: functions to preprocess the data and split it into train and validation subsets.
* **model.py**: functions to train and generate name proposals with language models.
* **train.py**: main to train a single model.
* **train_ensemble.py**: main to train an ensemble of models from the same architecture and data, varying only validation folds.
* **ranker.py**: basic functions and experimental stuff to ensemble model proposals.
* **outOfBagRanker.ipynb**: notebook used to ensemble models with the Out-Of-Bag + crawler-pasta strategy.
* **backtranslate.py**: main to backtranslate training data.

The rest of source files and notebooks contain mostly experimental stuff that didn't work out. Feel free to plunge into this mess if you feel like it.

Original data files, crawled data files and the source code for the Zara crawlers are not included in this repo.

"""
Functions for mining insights from texts.
"""

import re
from importlib import resources
from typing import Any, Generator

import en_core_web_sm
import joblib
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

# other settings
nlp = en_core_web_sm.load(disable=["ner"])


def prepocess_texts(
    texts: list[str], n_processes: int = 1, progress_bar: bool = True
) -> list[str]:
    """
    Preprocess texts by lemmatising tokens which is useful for sparse text representations.

    Parameters
    ----------
    texts : list[str]
        List of texts to preprocess.
    n_processes : int, default=1
        Number of processes to run in parallel with `nlp.pipe`.
    progress_bar : bool, default=True
        Flag to enable/disable a tqdm progress bar.

    Returns
    -------
    texts_clean : list[str]
        List of cleaned texts.
    """
    texts_clean = []
    is_valid = lambda t: not any(
        [
            t.is_digit,
            t.is_punct,
            t.is_stop,
            t.is_space,
            t.like_email,
            t.like_num,
            t.like_url,
        ]
    )
    for doc in tqdm(nlp.pipe(texts, n_process=n_processes), disable=not progress_bar):
        lemmas = [token.lemma_ for token in doc if is_valid(token)]
        texts_clean.append(" ".join(lemmas))
    return texts_clean


def yield_blocks(
    items: list[Any], size: int, overlap: int = 0
) -> Generator[list[Any], None, None]:
    """
    Split a list of items into (overlapping) blocks.

    Parameters
    ----------
    items : list[Any]
        List of items to split.
    size : int
        Desired number of items in each resulting block.
    overlap : int
        Desired overlap between two consecutive blocks in the number of items.

    Yields
    ------
    block : Generator[list[Any], None, None]
        Block of items of the desired size.
    """
    assert overlap < size, "Overlap must be smaller than size."
    for i in range(0, len(items), size - overlap):
        block = items[i : i + size]
        yield block


def predict_priorities(
    text: str, pipe: Pipeline, block_size: int = 200, dumping_factor: float = 1.1
) -> dict:
    """
    Predict priorities in a text using a classification model.

    This approach uses a multiclass classification model to approximate priorities
    by the amount of information linked to each class in the text. The higher the number
    of text blocks linked to a class, the higher its priority is.

    Parameters
    ----------
    text : str
        Raw text.
    pipe : Pipeline
        Sklearn Pipeline object. A classifier component must be called 'clf' and support `predict_proba` method.
    block_size : int, default=200
        Size of tetx block for splitting the text. This value must be close to the average text length of the training data.
    dumping_factor : float, default=1.1
        Value used to scale the minimum probability taken into account during calculations.
    Returns
    -------
    priorities : dict[str, float]
        Mapping from class names to scores ranging from 0 (lowest priority) to 1 (highest priority). It is guaranteed that the
        `priorities` contains at least 1 class with the priority of 1, unless all are 0.
    """
    with nlp.select_pipes(enable="tokenizer"):
        tokens = [token.text for token in nlp(text)]
    blocks = [" ".join(block) for block in yield_blocks(tokens, block_size)]
    blocks = prepocess_texts(blocks, progress_bar=False)
    probs = pipe.predict_proba(blocks)
    classes = pipe["clf"].classes_

    # ignore probabilities below the threshold
    uniform = 1 / len(classes)
    threshold = uniform * dumping_factor
    probs[probs < threshold] = 0

    # sum up classwise probability and renormalise scores to the range between 0 and 1.
    scores = probs.sum(axis=0)
    if scores.max() > 0:
        scores = (scores / scores.max()).tolist()
    else:
        scores = [0] * len(classes)
    priorities = dict(zip(classes, scores))
    return priorities


def detect_language(texts: list[str], min_threshold: float = 0.5) -> list[str | None]:
    import fasttext
    """
    Detect languages of texts.

    Parameters
    ----------
    texts : list[str]
        List of texts.
    min_threshold : float, default=0.5
        Ignore predictions with the probability below `min_threshold`.

    Returns
    -------
    labels : list[str | None]
        List of language codes, e.g., 'en', 'de', 'fr'. For labels predictions with probabilty below `min_threshold`, None is returned.
    """
    files = resources.files("cpe.data")
    model = fasttext.load_model(str(files.joinpath("lid.176.ftz")))
    texts = [re.sub(r"\s+", " ", text) for text in texts]
    labels = [
        labels[0].replace("__label__", "") if probs.item() > min_threshold else None
        for labels, probs in zip(*model.predict(texts))
    ]
    return labels


def load_model(file_path: str):
    """
    Load a joblib model/pipeline object.
    """
    return joblib.load(file_path)


def train_classifier(texts: list[str], labels: list[str]) -> Pipeline:
    """
    Train a classifier on the provided texts and labels.

    Parameters
    ----------
    texts : list[str]
        List of texts to train on.
    labels : list[str]
        List of labels corresponding to the texts.

    Returns
    -------
    pipe : Pipeline
        Trained sklearn Pipeline object.
    """
    vectorizer = TfidfVectorizer(max_features=30000, min_df=10, ngram_range=(2, 3), stop_words='english')
    classifier = SGDClassifier(loss="log_loss", random_state=42, class_weight="balanced", early_stopping=False)
    pipe = make_pipeline(vectorizer, classifier)
    pipe.fit(texts, labels)
    return pipe
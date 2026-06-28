import nltk
import spacy
from spacy.cli import download as spacy_download

def _ensure_nltk_data():
    try:
        nltk.data.find('corpora/names')
    except LookupError:
        nltk.download('names', quiet=True)

def _ensure_spacy_model(model_name="en_core_web_lg"):
    try:
        spacy.load(model_name)
    except OSError:
        spacy_download(model_name)

_ensure_nltk_data()
_ensure_spacy_model()

from .DLM import DLM
import nltk
import spacy
from spacy.cli.download import download as spacy_download

def _ensure_spacy_model(model_name="en_core_web_lg"):
    try:
        spacy.load(model_name)
    except OSError:
        spacy_download(model_name)
        
_ensure_spacy_model()

from .DLM import DLM
"""v0 tokenizer: lowercase + alphanumeric runs. No stemming, no stopwords.

Anserini's 0.184 MRR@10 baseline uses Porter stemming; going without costs
quality (we will measure exactly how much and decide with data, not faith).
"""
import re

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())

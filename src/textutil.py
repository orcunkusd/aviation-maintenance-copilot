"""Tokenizer shared by BM25 indexing and querying."""
import re


def tokenize(text):
    text = text.lower().replace("–", "-")  # AD numbers use en-dashes
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)

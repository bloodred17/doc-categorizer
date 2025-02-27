import re

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()


def preprocess_text_alphanumeric(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def preprocess_text_alpha(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(tokens)

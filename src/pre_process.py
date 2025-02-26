import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Sentence tokenizer
nltk.download('stopwords')  # Stopwords for filtering common words
nltk.download('averaged_perceptron_tagger')  # POS tagging (optional)
nltk.download('wordnet')  # Lemmatization
nltk.download('omw-1.4')  # WordNet support
nltk.download('punkt_tab')  # Missing resource

def clean_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)
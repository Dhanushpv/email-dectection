import string
from nltk.corpus import stopwords

# Shared preprocessing function used by model training and API

def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

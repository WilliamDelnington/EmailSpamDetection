import nltk
import re
import os
from Handlers.spam_email_patterns import url_patterns
from nltk.corpus import stopwords
import emoji
import pandas as pd
nltk.download('punkt')

def load_emails(path, label_types=["ham", "spam"]):
    """
    Load emails from a directory and return a list of email contents.
    """
    # Check if the directory exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The directory {path} does not exist.")
    
    # Specify the text and label arrays
    all_texts = []
    all_labels = []

    for label_type in label_types:
        folder_path = os.path.join(path, label_type)
        label = 0 if label_type == label_types[0] else 1
        
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='latin-1') as f:
                try:
                    all_texts.append(f.read())
                    all_labels.append(label_type)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    assert len(all_texts) == len(all_labels), "Mismatch between texts and labels."
    return all_texts, all_labels

def load_email_from_csv(path, text_column='text', label_column='label'):
    """
    Load emails from a CSV file and return a list of email contents.
    """
    df = pd.read_csv(path, encoding='latin-1')
    all_texts = df[text_column].tolist()
    all_labels = df[label_column].tolist()
    
    assert len(all_texts) == len(all_labels), "Mismatch between texts and labels."
    return all_texts, all_labels

def extract_urls(text):
    """
    Extract URLs from the text using regex patterns.
    """
    urls = re.findall(url_patterns, text)
    if urls:
        for i in range(len(urls)):
            urls[i] = re.sub(r'\s+', '', urls[i])
    return urls

def preprocess_text(text):
    """
    Preprocess the text by removing URLs, HTML tags, and non-alphanumeric characters.
    """
    # Remove URLs
    url = re.findall(url_patterns, text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize the text
    tokenized = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    filtered = [word for word in tokenized if word not in stop_words]
    
    return filtered

def remove_emojis(text):
    emoji_corpus = [emoji.demojize(c) for c in text]
    return emoji_corpus
import nltk
import re
import os
from spam_email_patterns import url_patterns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import emoji
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

def preprocess_text(text, remove_numbers=False, unncessary_words=None):
    """
    Preprocess the text by removing URLs, HTML tags, and non-alphanumeric characters.
    """
    # Remove URLs
    url = re.findall(url_patterns, text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

    if remove_numbers:
        # Remove numbers
        text = re.sub(r'\d+', '', text)

    if unncessary_words:
        # Remove unnecessary words
        for word in unncessary_words:
            text = re.sub(r'(?i)\b' + re.escape(word) + r'\b', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'\n', " ", text)

    # Tokenize the text
    tokenized = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    filtered = [word for word in tokenized if word not in stop_words]
    
    return filtered

def remove_emojis(text):
    emoji_corpus = [emoji.demojize(c) for c in text]
    return emoji_corpus

def visualize_wordcloud(text, title='Word Cloud'):
    """
    Visualize the word cloud of the text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def preprocess_and_vectorizing(
        text: list,
        lem_or_stem_method: WordNetLemmatizer | PorterStemmer, 
        vectorizer: CountVectorizer | TfidfVectorizer):

    if isinstance(lem_or_stem_method, WordNetLemmatizer):
        processed_text = " ".join([lem_or_stem_method.lemmatize(t) for t in text])
    elif isinstance(lem_or_stem_method, PorterStemmer):
        processed_text = " ".join([lem_or_stem_method.stem(t) for t in text])
    else:
        raise TypeError("Not a stemming or lemmatizing class")
    
    if not isinstance(vectorizer, (CountVectorizer | TfidfVectorizer)):
        raise TypeError("Object not a vectorizer")
    
    X = vectorizer.fit_transform(preprocess_text)

    return X
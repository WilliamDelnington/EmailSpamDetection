import nltk
import re
import os
from spam_email_patterns import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import emoji
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('punkt')
from typing import List
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from functools import partial

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
    text = str(text).lower()

    # Remove email
    text = re.sub(email_pattern, "EMAIL", str(text))

    text = re.sub(spaced_out_email_pattern, "EMAIL", str(text))

    # Remove URLs
    text = re.sub(url_patterns, "URL", str(text))

    # Remove phone number
    text = re.sub(phone_number_pattern, "PHONENUM", str(text))

    # Remove time
    text = re.sub(time_pattern, "TIME", str(text))

    # Remove date
    text = re.sub(date_patterns, "DATE", str(text))

    # Remove cost value
    text = re.sub(money_pattern, "VALUE", str(text))

    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

    text = re.sub("phonenum", "phone_number", str(text))

    if remove_numbers:
        # Remove numbers
        text = re.sub(r'\d+', '', str(text))

    if unncessary_words:
        # Remove unnecessary words
        for word in unncessary_words:
            text = re.sub(r'(?i)\b' + re.escape(word) + r'\b', '', str(text))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', str(text)).strip()

    text = re.sub(r'\n', " ", str(text))

    # Tokenize the text
    tokenized = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    filtered = [word for word in tokenized if word not in stop_words]
    
    return filtered

def remove_emojis(text):
    emoji_corpus = [emoji.demojize(c) for c in text]
    return emoji_corpus

def visualize_wordcloud(text: list, title='Word Cloud'):
    """
    Visualize the word cloud of the text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def lemmatizing(text: list):
    lemmatizer = WordNetLemmatizer()
    processed_text = " ".join([lemmatizer.lemmatize(t) for t in text])
    return processed_text

def stemming(text: list):
    stemmer = PorterStemmer()
    processed_text = " ".join([stemmer.stem(t) for t in text])
    return processed_text

def vectorizing(text: List[str], vectorizing_type: str, min_df=4):
    processed_type = vectorizing_type.lower().replace("-", "").replace(" ", "")
    if processed_type in ["count", "countvectorizer", "countvector", "countv"]:
        vectorizer = CountVectorizer(min_df=min_df)
    elif processed_type in ["tfidf", "tfidfvectorzier", "tfidfvector", "tfidfv"]:
        vectorizer = TfidfVectorizer(min_df=min_df)
    else:
        raise ValueError("Not a vectorizing type")
    X = vectorizer.fit_transform(text)
    return X

preprocession = partial(
    preprocess_text,
    remove_numbers=True
)

class EnronPreprocess:
    def __init__(self, enron: pd.DataFrame, name:str):
        self.__enron = enron
        self.name = name

    def preprocess_data(self, save=False):
        combined_input = self.__enron.apply(
            lambda x: f"{x['Subject']} {x['Body']}",
            axis=1
        )
        self.__preprocessed_data_X = combined_input.apply(preprocession)
        self.__preprocessed_data_y = self.__enron["Label"]
        if save:
            self.save_preprocessed_data(f"./csv/preprocessed_{self.name}_data.csv")
        return self.__preprocessed_data_X, self.__preprocessed_data_y

    def __get_counter(self):
        self.__word_list = list(itertools.chain.from_iterable(self.__preprocessed_data_X))
        self.__word_counter = Counter(self.__word_list)

    def visualize_wordcloud(self, minimum_occurance=4):
        self.__get_counter()

        if self.__preprocessed_data_X is None:
            return

        print(f"Total number of words: {len(self.__word_counter.keys())}")
        print(f"Total number of words that appear less than {minimum_occurance} times")
        print(len([key for key, value in self.__word_counter.items() if value < minimum_occurance]))
        
        visualize_wordcloud(self.__word_list)

    def visualize_bar_chart(
            self, 
            most_com=15,
            title="Top words frequencies",
            xlabel="Words",
            ylabel="Frequency"):
        self.__get_counter()

        most_common = self.__word_counter.most_common(most_com)

        words, counts = zip(*most_common)

        plt.figure(figsize=(10, 6))
        plt.bar(words, counts, color="skyblue")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    def save_preprocessed_data(self, path):
        """
        Save the preprocessed data to a CSV file.
        """
        if self.__preprocessed_data_X is None or self.__preprocessed_data_y is None:
            raise ValueError("Preprocessed data is not available.")
        
        df = pd.DataFrame({
            'Text': self.__preprocessed_data_X,
            'Label': self.__preprocessed_data_y
        })
        df.to_csv(path, index=False)
        print(f"Preprocessed data saved to {path}.")
import pandas as pd
import os
import nltk
import re
import csv
from datasets import load_dataset

# nltk.download('stopwords', download_dir="E:/nltk_data")
# nltk.download('punkt', download_dir="E:/nltk_data")

# enron_df = pd.read_csv('./Datasets/enron.csv')
# print(enron_df.head())

# print("--------------------------------------------------")

# spam_df = pd.read_csv('./Datasets/spam.csv', encoding='latin-1')
# print(spam_df.head())

# print("--------------------------------------------------")

# ling_df = pd.read_csv('./Datasets/ling.csv')
# print(ling_df.head())

# print("--------------------------------------------------")

# messages_df = pd.read_csv('./Datasets/messages.csv')
# print(messages_df.head())

# print("--------------------------------------------------")

# nazario_df = pd.read_csv('./Datasets/nazario.csv')
# print(nazario_df.head())

# print("--------------------------------------------------")

# spam_assassin_df = pd.read_csv('./Datasets/SpamAssasin.csv')
# print(spam_assassin_df.head())

all_data = []

ham_folder_paths = [
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron1/ham",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron2/ham",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron3/ham",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron4/ham",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron5/ham",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron6/ham"
]

spam_folder_paths = [
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron1/spam",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron2/spam",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron3/spam",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron4/spam",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron5/spam",
    "E:/Python Tests/AI/EmailSpamDetection/Datasets/enron6/spam"
]

# for ham_folder_path in ham_folder_paths:
#     for filename in os.listdir(ham_folder_path):
#         with open(os.path.join(ham_folder_path, filename), 'r', encoding='latin-1') as file:
#             content = file.read()
#             subject_match = re.search(r"Subject: (.*?)(?:\n|$)", content)
#             subject = subject_match.group(1) if subject_match else ""
#             all_data.append({"file_name": filename, "subject": subject, "text": content, "label": 0})  # 0 for ham

# for spam_folder_path in spam_folder_paths:
#     for filename in os.listdir(spam_folder_path):
#         with open(os.path.join(spam_folder_path, filename), 'r', encoding='latin-1') as file:
#             content = file.read()
#             subject_match = re.search(r"Subject: (.*?)(?:\n|$)", content)
#             subject = subject_match.group(1) if subject_match else ""
#             all_data.append({"file_name": filename, "subject": subject, "text": content, "label": 1})  # 1 for spam

# data = pd.DataFrame(all_data)
# data.to_csv('./Datasets/enron_combined.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL, escapechar="\\")

# enron_combined_df = pd.read_csv('./Datasets/enron_combined.csv')
# print(enron_combined_df.head())

combined_data_df = pd.read_csv('./Datasets/combined_data.csv')
print(combined_data_df.head())

ds = load_dataset("yxzwayne/email-spam-10k")

labels = ds["train"]["is_spam"]
text = ds["train"]["text"]

email_spam_10k_df = pd.DataFrame({"text": text, "label": labels})
email_spam_10k_df.to_csv("./datasets/email_spam_10k.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_ALL, escapechar="\\")
print("Transfer complete")
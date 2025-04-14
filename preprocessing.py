import pandas as pd
import os
import nltk
import re
import csv
from datasets import load_dataset

# nltk.download('stopwords', download_dir="E:/nltk_data")
# nltk.download('punkt', download_dir="E:/nltk_data")

all_data = []

enron_df = pd.read_csv('./Datasets/enron.csv')
# print("Enron Dataset:")
# print(enron_df["label"].value_counts())
# print(enron_df[enron_df["label"] == 1].head())

# print("--------------------------------------------------")

spam_df = pd.read_csv('./Datasets/spam.csv', encoding='latin-1')
# print("Spam Dataset:")
# print(spam_df["v1"].value_counts())
# print(spam_df[spam_df["v1"] == "spam"].head())

# print("--------------------------------------------------")

spam_dataset_df = pd.read_csv('./Datasets/spam_dataset.csv', encoding='latin-1')

ling_df = pd.read_csv('./Datasets/ling.csv')
# print("Ling Dataset:")
# print(ling_df["label"].value_counts())
# print(ling_df[ling_df["label"] == 1].head())

# print("--------------------------------------------------")

messages_df = pd.read_csv('./Datasets/messages.csv')
# print(messages_df.head())

# print("--------------------------------------------------")

# nazario_df = pd.read_csv('./Datasets/nazario.csv')
# print(nazario_df.head())

# print("--------------------------------------------------")

spam_assassin_df = pd.read_csv('./Datasets/SpamAssasin.csv')
# print("Spam Assassin Dataset:")
# print(spam_assassin_df["label"].value_counts())
# print(spam_assassin_df[spam_assassin_df["label"] == 1].head())

# print("--------------------------------------------------")

ceas_08_df = pd.read_csv('./Datasets/CEAS_08.csv')
# print("CEAS 08 Dataset:")
# print(ceas_08_df["label"].value_counts())
# print(ceas_08_df[ceas_08_df["label"] == 1].head())

phishing_df = pd.read_csv('./Datasets/Phishing_Email (2).csv')
# print("Phishing Dataset:")
# print(phishing_df["Email Type"].value_counts())
# print(phishing_df[phishing_df["Email Type"] == "Phishing Email"].head())

# print("--------------------------------------------------")

nigerian_fraud_df = pd.read_csv('./Datasets/Nigerian_Fraud.csv')
# print("Nigerian Fraud Dataset:")
# print(nigerian_fraud_df["label"].value_counts())

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

# combined_data_df = pd.read_csv('./Datasets/combined_data.csv')
# print(combined_data_df.head())

# ds = load_dataset("yxzwayne/email-spam-10k")

# labels = ds["train"]["is_spam"]
# text = ds["train"]["text"]

# email_spam_10k_df = pd.DataFrame({"text": text, "label": labels})
# email_spam_10k_df.to_csv("./datasets/email_spam_10k.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_ALL, escapechar="\\")
# print("Transfer complete")

# Example mappings
def extract_spam(df, spam_values, label_column='label'):
    return df[df[label_column].isin(spam_values)].copy()

enron_spam_df = extract_spam(enron_df, [1])
spam_spam_df = extract_spam(spam_df, ["spam"], label_column='Category')
spam_spam_2_df = extract_spam(spam_dataset_df, [1], label_column='is_spam')
# print(enron_spam_df)

def standardize(df, text_col='text', label_col='label', subject_col="subject"):
    if df.get(subject_col) is None:
        df["subject"] = ""
    return df.rename(columns={text_col: 'text', label_col: 'label', subject_col: 'subject'})[['text', 'label', "subject"]]

enron_spam_df = standardize(enron_spam_df, text_col='body')
spam_spam_df = standardize(spam_spam_df, text_col='Message', label_col='Category')
spam_spam_2_df = standardize(spam_spam_2_df, text_col='message_content', label_col='is_spam')

merge = pd.concat([enron_spam_df, spam_spam_df, spam_spam_2_df], ignore_index=True)
# print(enron_spam_df)
print(merge)
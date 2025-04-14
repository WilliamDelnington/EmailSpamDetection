from spam_email_patterns import adult_content_patterns
import pandas as pd
import re

df = pd.read_csv("./Datasets/merged_spam_dataset.csv")

def classify_scam(text):
    # Check for adult content patterns
    for pattern in adult_content_patterns:
        if re.search(pattern, text):
            return 1  # Spam
    return 0  # Not spam

import pandas as pd

# Function to determine data type
def get_type(value):
    if pd.isna(value):
        return "NoneType"
    try:
        int_val = int(value)
        return "int"
    except ValueError:
        try:
            float_val = float(value)
            return "float"
        except ValueError:
            return "str"

# Apply to specific column
types = df['age'].apply(get_type)

# Count types
type_counts = types.value_counts()

print(type_counts)

# df["is_scam"] = df["text"].apply(classify_scam)
# print(df["is_scam"].value_counts())
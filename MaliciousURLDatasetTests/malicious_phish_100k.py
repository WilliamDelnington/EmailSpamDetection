from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("./malicious_phish_preprocessed.csv")

# We will create a new sub-dataset from the original dataset that contains only 100k samples.
y = df["type"]
X = df.drop(columns=["type"])

print(X, y)

X_subset, _, y_subset, _ = train_test_split(X, y, train_size=200000, stratify=y, random_state=42)

pd.concat([X_subset, y_subset], axis=1).to_csv("./malicious_phish_preprocessed_100k.csv", index=False)
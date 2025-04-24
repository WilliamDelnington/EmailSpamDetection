from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn as nn

class SupportVectorMachine:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        # Create a Support Vector Machine classifier
        self.model = svm.SVC(kernel='linear')

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

class NaiveBayes:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        
        # Create a Naive Bayes classifier
        self.model = GaussianNB()

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

class RandomForest:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        # Create a Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100)

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

class KNearestNeighbors:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        # Create a K-Nearest Neighbors classifier
        self.model = KNeighborsClassifier(n_neighbors=5)

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

class RecurrentNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(RecurrentNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out[:, -1, :])
        return out
    
class LogisticRegressionModel:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        # Create a Logistic Regression classifier
        self.model = LogisticRegression()

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))
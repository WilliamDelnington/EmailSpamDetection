from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

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
        # Check if the model has been trained
        if (self.X_train is None or
            self.X_test is None or
            self.y_train is None or
            self.y_test is None):
            raise ValueError("Model has not been trained yet. Call train() before evaluate().")
        
        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

    def visualize_confusion_matrix(self):
        # Check if the confusion matrix has been computed
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix has not been computed. Call evaluate() before visualize_confusion_matrix().")
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.model.classes_)

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

class GaussianNaiveBayes:
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
        # Check if the model has been trained
        if (self.X_train is None or
            self.X_test is None or
            self.y_train is None or
            self.y_test is None):
            raise ValueError("Model has not been trained yet. Call train() before evaluate().")
        
        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))

        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

    def visualize_confusion_matrix(self):
        # Check if the confusion matrix has been computed
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix has not been computed. Call evaluate() before visualize_confusion_matrix().")
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.model.classes_)

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

class MultinomialNaiveBayes:
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

        # Create a Multinomial Naive Bayes classifier
        self.model = MultinomialNB()

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Check if the model has been trained
        if (self.X_train is None or
            self.X_test is None or
            self.y_train is None or
            self.y_test is None):
            raise ValueError("Model has not been trained yet. Call train() before evaluate().")

        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

    def visualize_confusion_matrix(self):
        # Check if the confusion matrix has been computed
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix has not been computed. Call evaluate() before visualize_confusion_matrix().")
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.model.classes_)

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

class BernoulliNaiveBayes:
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

        # Create a Bernoulli Naive Bayes classifier
        self.model = BernoulliNB()

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Check if the model has been trained
        if (self.X_train is None or
            self.X_test is None or
            self.y_train is None or
            self.y_test is None):
            raise ValueError("Model has not been trained yet. Call train() before evaluate().")

        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

    def visualize_confusion_matrix(self):
        # Check if the confusion matrix has been computed
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix has not been computed. Call evaluate() before visualize_confusion_matrix().")
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.model.classes_)

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

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
        # Check if the model has been trained
        if (self.X_train is None or
            self.X_test is None or
            self.y_train is None or
            self.y_test is None):
            raise ValueError("Model has not been trained yet. Call train() before evaluate().")

        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

    def visualize_confusion_matrix(self):
        # Check if the confusion matrix has been computed
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix has not been computed. Call evaluate() before visualize_confusion_matrix().")
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.model.classes_)

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

class DecisionTree:
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

        # Create a Decision Tree classifier
        self.model = DecisionTreeClassifier()

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Check if the model has been trained
        if (self.X_train is None or
            self.X_test is None or
            self.y_train is None or
            self.y_test is None):
            raise ValueError("Model has not been trained yet. Call train() before evaluate().")

        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

    def visualize_confusion_matrix(self):
        # Check if the confusion matrix has been computed
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix has not been computed. Call evaluate() before visualize_confusion_matrix().")
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.model.classes_)

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

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
        # Check if the model has been trained
        if (self.X_train is None or
            self.X_test is None or
            self.y_train is None or
            self.y_test is None):
            raise ValueError("Model has not been trained yet. Call train() before evaluate().")

        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))

        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

    def visualize_confusion_matrix(self):
        # Check if the confusion matrix has been computed
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix has not been computed. Call evaluate() before visualize_confusion_matrix().")
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.model.classes_)

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

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
        # Check if the model has been trained
        if (self.X_train is None or
            self.X_test is None or
            self.y_train is None or
            self.y_test is None):
            raise ValueError("Model has not been trained yet. Call train() before evaluate().")

        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

    def visualize_confusion_matrix(self):
        # Check if the confusion matrix has been computed
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix has not been computed. Call evaluate() before visualize_confusion_matrix().")
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.model.classes_)

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

class LongShortTermMemory(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, n_layers, dropout):
        super(LongShortTermMemory, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)
    
class AdaBoost:
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

        # Create an AdaBoost classifier
        self.model = AdaBoostClassifier(n_estimators=100)

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Check if the model has been trained
        if (self.X_train is None or
            self.X_test is None or
            self.y_train is None or
            self.y_test is None):
            raise ValueError("Model has not been trained yet. Call train() before evaluate().")

        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(classification_report(self.y_test, y_pred))
        
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

    def visualize_confusion_matrix(self):
        # Check if the confusion matrix has been computed
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix has not been computed. Call evaluate() before visualize_confusion_matrix().")
        
        # Create a ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.model.classes_)

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
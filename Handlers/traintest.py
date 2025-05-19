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
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA, TruncatedSVD
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import traceback as trb

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

models = [
    svm.SVC(kernel="linear"),
    svm.SVC(kernel="rbf"),
    MultinomialNB(),
    BernoulliNB(),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    LogisticRegression(),
    KNeighborsClassifier()
]

class EvaluateError(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)

class ClassificationModel:
    def __init__(self, model):
        """
        Initialize the classifier with a sklearn model

        Parameters:
        - model: The model to be used for training
        - X: The sample of data
        - y: The label of data
        - test_size: The test size ratio to split into the training and testing sets.
        - random_state: Set the random state when splitting training and testing model.
        """
        if not isinstance(model, BaseEstimator):
            raise TypeError("Classification model must be built from sklearn")
        self.model = model

        if isinstance(self.model, svm.SVC):
            self.name = "SVM"
        elif isinstance(self.model, GaussianNB):
            self.name = "Gaussian Naive Bayes"
        elif isinstance(self.model, MultinomialNB):
            self.name = "Multinomial Naive Bayes"
        elif isinstance(self.model, BernoulliNB):
            self.name = "Bernoulli Naive Bayes"
        elif isinstance(self.model, RandomForestClassifier):
            self.name = "Random Forest"
        elif isinstance(self.model, DecisionTreeClassifier):
            self.name = "Decision Tree"
        elif isinstance(self.model, LogisticRegression):
            self.name = "Logistic Regression"
        elif isinstance(self.model, KNeighborsClassifier):
            self.name = "K-nearest Neighbors"
        elif isinstance(self.model, AdaBoostClassifier):
            self.name = "AdaBoost"
        else:
            self.name = ""

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def train(self, X, y, random_state=42, test_size=0.2, partial=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # try:
        #     self.__fitting_model(self.X_train, self.y_train, partial=partial)

        # except (TypeError, AttributeError):
        #     print("Dense data is required")
        #     print("Converting to dense array")

        #     func = FunctionTransformer(lambda x: x.todense(), accept_sparse=True)
        #     self.X_train = func.transform(self.X_train)
        #     self.X_test = func.transform(self.X_test)

        #     try:
        #         self.__fitting_model(self.X_train, self.y_train, partial=partial)

        #     except (TypeError):
        #         print("Numpy array required")
        #         print("Converting to numpy array")

        #         try:
        #             self.X_train = np.array(self.X_train)
        #             self.X_test = np.array(self.X_test)

        #             self.__fitting_model(self.X_train, self.y_train, partial=partial)
        #         except MemoryError:
        #             svd = TruncatedSVD(n_components=100)
        #             self.X_train = svd.fit_transform(self.X_train)
        #             self.X_test = svd.transform(self.X_test)
        #             self.__fitting_model(self.X_train, self.y_train, partial=partial)

        #     except Exception as e:
        #         print("Error in fitting model:", trb.print_exc())

        if isinstance(self.model, GaussianNB):
            try:
                func = FunctionTransformer(lambda x: x.todense(), accept_sparse=True)
                X_temp = func.transform(self.X_train)
                self.X_test = func.transform(self.X_test)

                self.__fitting_model(X_temp, self.y_train, partial=partial)
            except (TypeError, AttributeError):
                func = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
                X_temp = np.array(self.X_train)
                self.X_test = np.array(self.X_test)

                self.__fitting_model(X_temp, self.y_train, partial=partial)

        else:
            self.__fitting_model(self.X_train, self.y_train, partial=partial)


    def evaluate(self):
        if (self.X_train is None or
            self.y_train is None or 
            self.X_test is None or
            self.y_test is None):
            raise EvaluateError("The model hasn't been trained yet.")
        
        y_pred = self.model.predict(self.X_test)

        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)

        return classification_report(self.y_test, y_pred)
    
    def draw_confusion_matrix(self):
        if not hasattr(self, "confusion_matrix"):
            raise ValueError("Model is not evaluated.")
        
        disp = ConfusionMatrixDisplay(self.confusion_matrix, display_labels=self.model.classes_)

        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{self.name} Confusion Matrix")
        plt.show()

    def __fitting_model(self, X, y, partial=False):
        if not partial:
            self.model.fit(X, y)
        else:
            self.model.partial_fit(X, y)
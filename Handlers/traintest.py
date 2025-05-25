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
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from keras.src.models import Sequential
from keras.src.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout, LSTM, Bidirectional
from keras.src.layers import TextVectorization
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import traceback as trb
import pandas as pd

class PreprocessAndTrainWithRNN(nn.Module):
    # def __init__(self, vocab_size, embedding_size, num_filters, filter_sizes, hidden_size, output_size, dropout=0.2):
    def __init__(self):
        self.model = Sequential()

    def load_data(self, 
                  data, 
                  input_features:list=None, 
                  output_feature:str=None
                  ):
        """
        Load and preprocess the data.
        Parameters:
        - data: The input data to be processed.
        - features: Optional, specific features to be used from the data.
        """
        # Implement if data is a DataFrame object
        if isinstance(data, pd.DataFrame):
            if input_features is None and output_feature is None:
                print("The input features are not specified. The last column will be used as the output feature, while the rest will be used as input features.")
                self.X = data.columns[:-1].tolist()
                self.y = data.columns[-1]
            elif input_features is None:
                self.X = data[input_features].tolist()
                self.y = data[[f for f in data.columns if f not in input_features]].tolist()
            elif output_feature is None:
                self.X = data.drop(columns=[output_feature]).tolist()
                self.y = data[output_feature].tolist()
            else:
                self.X = data[input_features].tolist()
                self.y = data[output_feature].tolist()

        elif isinstance(data, (np.ndarray, list)):
            self.X = data[:, :-1]
            self.y = data[:, -1]

        else:
            raise TypeError("Unsupported data type. Please provide a DataFrame, numpy array, or list.")
        
    def split(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        Parameters:
        - test_size: The proportion of the dataset to include in the test split.
        - random_state: Controls the shuffling applied to the data before applying the split.
        """
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            raise ValueError("Data has not been loaded. Call load_data() first.")
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def preprocess(self, max_features=10000, sequence_length=100):
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.vectorizer = TextVectorization(
            max_tokens=max_features, 
            output_mode='int', 
            output_sequence_length=sequence_length
        )
        self.vectorizer.adapt(self.x_train)


    def build(self, max_features=10000, 
              embedding_size=128, 
              num_filters=64, 
              filter_sizes=3, 
              hidden_size=128, 
              dropout_rate=0.2,
              epochs=10,
              batch_size=32):
        
        self.model.add(self.vectorizer)
    
        self.model.add(Embedding(
            input_dim=max_features, 
            output_dim=embedding_size
        ))
        self.model.add(Conv1D(
            filters=num_filters, 
            kernel_size=filter_sizes, 
            activation='relu',
            padding='valid',
            strides=1
        ))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'precision', 'recall']
        )

        self.model.fit(
            self.x_train, 
            self.y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(self.x_test, self.y_test)
        )
    
    def evaluate(self):
        if not hasattr(self, 'x_test') or not hasattr(self, 'y_test'):
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = (y_pred > 0.5).astype(int)

        report = classification_report(self.y_test, y_pred_classes)
        print(report)
    

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
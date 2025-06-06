import scipy.sparse
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    HistGradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (
    LogisticRegression, 
    SGDClassifier, 
    Perceptron,
    PassiveAggressiveClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from keras.src.models import Sequential
from keras.src.layers import (
    Dense, 
    Conv1D, 
    GlobalMaxPooling1D, 
    Embedding, 
    Dropout, 
    LSTM, 
    Bidirectional, 
    GRU, 
    MaxPooling1D
)
from keras.src.layers import TextVectorization
from keras.src.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau
)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from transformers.trainer_utils import PredictionOutput
from datasets import load_dataset, DatasetDict, Dataset, Features, Value, ClassLabel
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import traceback as trb
import pandas as pd
import scipy
import joblib
import json
import math
import os

model_parameters = {
    "SVM": {
        "kernel": ["linear", "rbf", "sigmoid"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
    },
    "GaussianNB": {
        "var_smoothing": [1e-9, 1e-8, 1e-7],
    },
    "MultinomialNB": {
        "alpha": [0.1, 0.5, 1.0],
        "fit_prior": [True, False],
    },
    "BernoulliNB": {
        "alpha": [0.1, 0.5, 1.0],
        "fit_prior": [True, False],
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "DecisionTree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0],
    },
    "LogisticRegression": {
        "penalty": ["l2", "elasticnet", None],
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "liblinear", "sag", "saga"],
    },
    "KNeighbors": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    },
    "SGDClassifier": {
        "loss": ["hinge", "log_loss", "squared_hinge"],
        "penalty": ["l2", "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    },
    "HistGradientBoosting": {
        "n_estimators": [100, 200],
        "max_depth": [5, 7],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [5, 10, 20],
    },
    "Perceptron": {
        "penalty": ["l2", "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [1000, 2000, 3000],
        "tol": [1e-3, 1e-4, 1e-5],
    },
    "PassiveAggressive": {
        "C": [0.1, 1, 10],
        "loss": ["hinge", "squared_hinge"],
        "max_iter": [500, 1000, 2000],
        "tol": [1e-3, 1e-4, 1e-5],
    },
    "MLPClassifier": {
        "activation": ["relu", "identity", "logistic", "tanh"],
        "learning_rate": [0.001, 0.01, 0.1],
        "solver": ["lbfgs", "sgd", "adam"]
    },
    "ExtraTreesClassifier": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 15]
    },
    "XGBClassifier": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.001, 0.01, 0.1]
    }
}

class NeuralNetworkClassifier(ABC):
    def __init__(self, 
                 data_name:str, 
                 model_name:str, 
                 max_features:int, 
                 input_length:int):
        self.model = Sequential()
        self.epochs = 10
        self.history = None
        self.data_name = data_name
        self.model_name = model_name
        self.max_features = max_features
        self.input_length = input_length

    def load_data(self, X, y):
        """
        Load and preprocess the data.
        Parameters:
        - X: The input samples and features to be processed.
        - y: The input label from each sample of the dataset
        """
        self.X = X
        self.y = y
        if isinstance(self.X, (pd.DataFrame, pd.Series)):
            self.X = self.X.astype(str).values
        if isinstance(self.y, (pd.DataFrame, pd.Series)):
            self.y = self.y.astype(np.int32).values

    def split(self, test_size=0.1, valid_size=0.1, random_state=42):
        """
        Split the data into training and testing sets.
        Parameters:
        - test_size: The proportion of the dataset to include in the test part.
        - valid_size: The proportion of the dataset to include in the validation part.
        - random_state: Controls the shuffling applied to the data before applying the split.
        """
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            raise ValueError("Data has not been loaded. Call load_data() first.")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        if valid_size > 0:
            val_ratio = valid_size / (1 - test_size)

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=val_ratio, random_state=random_state
            )

    def reduce_dim(self, input, num_features=100):
        try:
            l = self.X.shape
            print(l)
        except Exception as e:
            print(f"An error when getting the shape: {trb.print_exc()}")
            l = len(self.X[0])
        svd = TruncatedSVD(n_components=num_features)
        self.X_train = svd.fit_transform(input)
        
    def vectorizing(self, data_input=None, print_shape=True):
        self.vectorizer = TextVectorization(
            max_tokens=self.max_features,
            output_mode="int",
            output_sequence_length=self.input_length
        )

        self.vectorizer.adapt(self.X_train)

        if data_input is None:
            vectorized = self.vectorizer(self.X_train)
        else:
            vectorized = self.vectorizer(data_input)
        
        if print_shape:
            print(f"Shape of the input data is {vectorized.shape}")
        
        return vectorized
    
    def get_callbacks(
        self,
        callback_methods=["early-stop"],
        early_stopping_monitor="val_accuracy",
        epoch_limitation=5,
        min_delta=0.001,
        save_path="./best_model.h5"
    ):
        """
        Setup callbacks for the model whether the model stops improving.

        Note that early-stopping must always come first.
        """
        self.callbacks = []
        for method in callback_methods:
            if method.lower() in [
                "earlystopping", 
                "earlystop", 
                "early stop", 
                "early stopping"
            ]:
                early_stopping = EarlyStopping(
                    monitor=early_stopping_monitor,
                    patience=epoch_limitation,
                    min_delta=min_delta,
                    mode="max",
                    verbose=1,
                    restore_best_weights=True
                )

                self.callbacks.append(early_stopping)

            elif method.lower() in [
                "model checkpoint", 
                "modelcheckpoint"
            ]:
                if len(self.callbacks) == 0:
                    raise ValueError("Early-stopping must always comes first.")
                model_checkpoint = ModelCheckpoint(
                    save_path,
                    monitor=early_stopping_monitor,
                    save_best_only=True,
                    mode="max",
                    verbose=1
                )

                self.callbacks.append(model_checkpoint)

            elif method.lower() in [
                "reducelr", 
                "reduce learning rate", 
                "reducelearningrate",
                "reducelronplateau",
                "reduce learning rate on plateau"
            ]:
                if len(self.callbacks) == 0:
                    raise ValueError("Early-stopping must always comes first.")
                reducelr = ReduceLROnPlateau(
                    monitor=early_stopping_monitor,
                    factor=0.5,
                    patience=epoch_limitation-3,
                    min_lr=1e-6,
                    verbose=1
                )

                self.callbacks.append(reducelr)

            else:
                raise ValueError("""
                    Unknown callback method. The available methods are
                    EarlyStopping, ModelCheckpoint and ReduceLROnPlateau""")

    @abstractmethod
    def build(self):
        pass

    def plot_training_validation_accuracy(
            self,
            figsize=(10, 6),
            plot_xlabel="Epochs",
            plot_ylabel="Accuracy",
            save_plot=True,
            parent_folder="./figs/"):
        if self.history is None:
            raise ValueError("Model is not trained. Call build() method before plotting.")
        accuracy = self.history.history["accuracy"]
        plt.figure(figsize=figsize)
        plt.plot(
            range(1, self.epochs + 1),
            accuracy,
            color="blue", 
            linestyle="--", 
            linewidth=2, 
            label="Training Accuracy"
        )
        if hasattr(self, "X_val"):
            val_accuracy = self.history.history["val_accuracy"]
            plt.plot(
                range(1, self.epochs + 1),
                val_accuracy,
                color="red", 
                linestyle="--", 
                linewidth=2, 
                label="Validation Accuracy"
            )
        plt.title(f"{self.model_name} Train-Val Accuracy Results for {self.data_name}")
        plt.xlabel(xlabel=plot_xlabel)
        plt.ylabel(ylabel=plot_ylabel)
        plt.grid(True, alpha=0.3)
        if save_plot:
            plt.savefig(os.path.join(parent_folder, f"{self.model_name}_{self.data_name}.jpg"))
        plt.show()

    def evaluate(self, detailed=True):
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)

        self.confusion_matrix = confusion_matrix(self.y_test, y_pred_classes)
        if detailed:
            self.metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred_classes),
                "weighted_precision": precision_score(self.y_test, y_pred_classes, average='weighted'),
                "wighted_recall": recall_score(self.y_test, y_pred_classes, average='weighted'),
                "weighted_f1": f1_score(self.y_test, y_pred_classes, average='weighted'),
                "macro_precision": precision_score(self.y_test, y_pred_classes, average='macro'),
                "macro_recall": recall_score(self.y_test, y_pred_classes, average='macro'),
                "macro_f1": f1_score(self.y_test, y_pred_classes, average='macro'),
                "roc_auc": roc_auc_score(self.y_test, y_pred_classes)
            }
            detailed_metrics = {
                "dataset": self.data_name,
                "model": self.model_name,
                "metrics": self.metrics,
                "confusion_matrix": self.confusion_matrix,
                "epochs": self.epochs
            }
            return detailed_metrics

        return classification_report(self.y_test, y_pred)

class ConvolutionalNNClassifier(NeuralNetworkClassifier):
    # def __init__(self, vocab_size, embedding_size, num_filters, filter_sizes, hidden_size, output_size, dropout=0.2):
    def __init__(self, data_name, model_name="CNN", max_features=5000, input_length=200):
        super().__init__(data_name, model_name, max_features, input_length)

    def build(self,
              embedding_size=128, 
              num_filters=[64],
              conv_layer_num=1,
              kernel_sizes=[5], 
              hidden_size=64,
              max_pooling=False,
              pooling_sizes=[2],
              pooling_dropout=False, 
              pooling_dropout_rate=0.2,
              dense_dropout=False,
              dense_dropout_rate=0.5,
              epochs=10,
              batch_size=32,
              callback_methods=["early stopping"],
              epoch_limitation=7,
              save_model=True):

        assert len(kernel_sizes) == conv_layer_num, "the len of filter_sizes parameters must match the number of hidden layer"
        assert len(num_filters) == conv_layer_num, "the len of num_filters parameters must match the number of hidden layer"
        
        self.model.add(self.vectorizer)
    
        self.model.add(Embedding(
            input_dim=self.max_features, 
            output_dim=embedding_size,
            input_length=self.input_length
        ))

        for i in range(conv_layer_num):
            self.model.add(Conv1D(
                filters=num_filters[i], 
                kernel_size=kernel_sizes[i], 
                activation='relu'
            ))

            if max_pooling:
                assert len(pooling_sizes) == conv_layer_num
                assert pooling_sizes[conv_layer_num - 1] == 0

                if pooling_sizes[i] != 0:
                    self.model.add(MaxPooling1D(pooling_sizes[i]))

        self.model.add(GlobalMaxPooling1D())

        if pooling_dropout:
            self.model.add(Dropout(pooling_dropout_rate))

        self.model.add(Dense(hidden_size, activation='relu'))

        if dense_dropout:
            self.model.add(Dropout(dense_dropout_rate))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'precision', 'recall']
        )

        self.get_callbacks(
            callback_methods=callback_methods,
            epoch_limitation=epoch_limitation
        )

        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=(None if len(self.callbacks) == 0 else self.callbacks),
            validation_data=(self.X_val, self.y_val)
        )

        stopped_epoch = self.callbacks[0].stopped_epoch

        self.epochs = stopped_epoch + 1 if stopped_epoch != 0 else epochs

        if save_model:
            self.model.save(f"./models/Classify_{self.data_name}_CNN_model.h5")

class RecurrentNNClassifier(NeuralNetworkClassifier):
    def __init__(self, data_name, model_name="RNN", max_features=5000, input_length=200):
        super().__init__(data_name, model_name, max_features, input_length)

    def build(self,
              embedding_size=64, 
              units=64,
              hidden_layer_num=1,
              hidden_sizes=[128],
              lstm=True,
              pooling_dropout=False, 
              pooling_dropout_rate=0.2,
              dense_dropout=False,
              dense_dropout_rate=0.5,
              bidirectional=False,
              epochs=10,
              batch_size=32,
              epoch_limitation=7,
              callback_methods=["early stopping"],
              save_model=True):
        
        assert len(hidden_sizes) == hidden_layer_num, "the size of hidden_sizes parameters must match the number of hidden layer"
        
        self.model.add(self.vectorizer)
    
        self.model.add(Embedding(
            input_dim=self.max_features, 
            output_dim=embedding_size,
            input_length=self.input_length
        ))

        for i in range(hidden_layer_num):
            if bidirectional:
                if i == hidden_layer_num - 1:
                    if lstm:
                        self.model.add(Bidirectional(LSTM(hidden_sizes[i])))
                    else:
                        self.model.add(Bidirectional(GRU(hidden_sizes[i])))
                else:
                    if lstm:
                        self.model.add(Bidirectional(LSTM(hidden_sizes[i], return_sequences=True)))
                    else:
                        self.model.add(Bidirectional(GRU(hidden_sizes[i], return_sequences=True)))
            else:
                if i == hidden_layer_num - 1:
                    if lstm:
                        self.model.add(LSTM(hidden_sizes[i]))
                    else:
                        self.model.add(GRU(hidden_sizes[i]))
                else:
                    if lstm:
                        self.model.add(LSTM(hidden_sizes[i], return_sequences=True))
                    else:
                        self.model.add(GRU(hidden_sizes[i], return_sequences=True))

        if pooling_dropout:
            self.model.add(Dropout(pooling_dropout_rate))

        self.model.add(Dense(units, activation='relu'))

        if dense_dropout:
            self.model.add(Dropout(dense_dropout_rate))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'precision', 'recall']
        )

        self.get_callbacks(
            callback_methods=callback_methods,
            epoch_limitation=epoch_limitation
        )

        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=(None if len(self.callbacks) == 0 else self.callbacks),
            validation_data=(self.X_val, self.y_val)
        )

        stopped_epoch = self.callbacks[0].stopped_epoch

        self.epochs = stopped_epoch + 1 if stopped_epoch != 0 else epochs

        if save_model:
            self.model.save(f"./models/Classify_{self.data_name}_RNN_model.h5")

class ArtificialNNClassifier(NeuralNetworkClassifier):
    def __init__(self, data_name, model_name="ANN", max_features=5000, input_length=200):
        super().__init__(data_name, model_name, max_features, input_length)

    def build(self,
              hidden_layer_num=2,
              hidden_layer_sizes=[64, 64],
              embedding_size=64,
              callback_methods=["early stopping"],
              epochs=10,
              batch_size=32,
              epoch_limitation=7,
              save_model=True):
        assert len(hidden_layer_sizes) == hidden_layer_num

        self.model.add(self.vectorizer)

        self.model.add(Embedding(
            input_dim=self.max_features,
            output_dim=embedding_size
        ))

        self.model.add(GlobalMaxPooling1D())

        for i in range(hidden_layer_num):
            self.model.add(Dense(hidden_layer_sizes[i], activation="relu"))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'precision', 'recall']
        )

        self.get_callbacks(
            callback_methods=callback_methods,
            epoch_limitation=epoch_limitation
        )

        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=(None if len(self.callbacks) == 0 else self.callbacks),
            validation_data=(self.X_val, self.y_val)
        )

        stopped_epoch = self.callbacks[0].stopped_epoch

        self.epochs = stopped_epoch + 1 if stopped_epoch != 0 else epochs

        if save_model:
            self.model.save(f"./models/Classify_{self.data_name}_ANN_model.h5")

def get_classification_models(X):
    return [
        SVC(),
        MultinomialNB(),
        BernoulliNB(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        AdaBoostClassifier(),
        LogisticRegression(n_jobs=4),
        KNeighborsClassifier(n_jobs=4),
        SGDClassifier(),
        # HistGradientBoostingClassifier(
        #     early_stopping=True, 
        #     n_iter_no_change=6,
        #     max_bins=199,
        #     ),
        Perceptron(),
        PassiveAggressiveClassifier(),
        MLPClassifier(hidden_layer_sizes=(150, 75), early_stopping=True),
        ExtraTreesClassifier(),
        XGBClassifier()
    ]

def get_nn_classification_models(dataset_name):
    return [
        ConvolutionalNNClassifier(dataset_name),
        RecurrentNNClassifier(dataset_name),
        ArtificialNNClassifier(dataset_name)
    ]

class EvaluateError(Exception):
    def __init__(self, *args):
        super(Exception, self).__init__(*args)

class ClassificationModel:
    def __init__(self, model, data_name:str):
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
        self.data_name = data_name

        if isinstance(self.model, SVC):
            self.model_name = "SVM"
            self.model_parameters = model_parameters["SVM"]
        elif isinstance(self.model, GaussianNB):
            self.model_name = "Gaussian Naive Bayes"
            self.model_parameters = model_parameters["GaussianNB"]
        elif isinstance(self.model, MultinomialNB):
            self.model_name = "Multinomial Naive Bayes"
            self.model_parameters = model_parameters["MultinomialNB"]
        elif isinstance(self.model, BernoulliNB):
            self.model_name = "Bernoulli Naive Bayes"
            self.model_parameters = model_parameters["BernoulliNB"]
        elif isinstance(self.model, RandomForestClassifier):
            self.model_name = "Random Forest"
            self.model_parameters = model_parameters["RandomForest"]
        elif isinstance(self.model, DecisionTreeClassifier):
            self.model_name = "Decision Tree"
            self.model_parameters = model_parameters["DecisionTree"]
        elif isinstance(self.model, LogisticRegression):
            self.model_name = "Logistic Regression"
            self.model_parameters = model_parameters["LogisticRegression"]
        elif isinstance(self.model, KNeighborsClassifier):
            self.model_name = "K-nearest Neighbors"
            self.model_parameters = model_parameters["KNeighbors"]
        elif isinstance(self.model, AdaBoostClassifier):
            self.model_name = "AdaBoost"
            self.model_parameters = model_parameters["AdaBoost"]
        elif isinstance(self.model, SGDClassifier):
            self.model_name = "Stochastic Gradient Descent"
            self.model_parameters = model_parameters["SGDClassifier"]
        elif isinstance(self.model, HistGradientBoostingClassifier):
            self.model_name = "Gradient Boosting"
            self.model_parameters = model_parameters["HistGradientBoosting"]
        elif isinstance(self.model, Perceptron):
            self.model_name = "Perceptron"
            self.model_parameters = model_parameters["Perceptron"]
        elif isinstance(self.model, PassiveAggressiveClassifier):
            self.model_name = "Passive-Aggressive"
            self.model_parameters = model_parameters["PassiveAggressive"]
        elif isinstance(self.model, MLPClassifier):
            self.model_name = "Multi-layer Perceptron"
            self.model_parameters = model_parameters["MLPClassifier"]
        elif isinstance(self.model, ExtraTreesClassifier):
            self.model_name = "Extra Trees Classifier"
            self.model_parameters = model_parameters["ExtraTreesClassifier"]
        elif isinstance(self.model, XGBClassifier):
            self.model_name = "XGBoost Classifier"
            self.model_parameters = model_parameters["XGBClassifier"]
        else:
            self.model_name = ""
            self.model_parameters = {}

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.through_validation = False
        self.confusion_matrix = None
        self.grid_searching = False
        self.normal_training = False
        self.epoch_training = False

    def __split_train_val_test(self, X, y, test_size=0.2, valid_size=0, random_state=42):
        """
        Split the data into training, validation, and testing sets.
        
        Parameters:
        - X: The input features.
        - y: The target labels.
        - test_size: The proportion of the dataset to include in the test split.
        - valid_size: The proportion of the training set to include in the validation split.
        - random_state: Controls the shuffling applied to the data before applying the split.
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if not 0 <= valid_size < 1:
            raise ValueError("valid_size must be between 0 and 1")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if valid_size > 0:
            val_ratio = valid_size / (1 - test_size)

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=val_ratio, random_state=random_state
            )

    def train(self, X, y, random_state=42, test_size=0.2, valid_size=0, partial=False, save_model=False):
        if not self.__is_valid_input(X):
            raise TypeError("Input data must be a numpy array, pandas DataFrame, list, or pandas Series.")
        if self.__is_trained():
            print("The model has already been trained. This process will overwrite the previous training.")


        self.__split_train_val_test(X, y, test_size=test_size, valid_size=valid_size, random_state=random_state)

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
            try:
                self.__fitting_model(self.X_train, self.y_train, partial=partial)
            except TypeError:
                self.X_train = self.X_train.toarray()
                self.__fitting_model(self.X_train, self.y_train, partial=partial)
        self.normal_training = True

        if not partial:
            del self.X_train, self.y_train

        if save_model:
            try:
                self.__save_model(self.model, f"./models/normal/{self.model_name}_{self.data_name}_normal.joblib")
            except Exception as e:
                print(f"Error saving model: {e}")
                trb.print_exc()

    def train_with_epochs(self, X, y, epochs=10, random_state=42, test_size=0.2, valid_size=0, save_model=False):
        if not self.__is_valid_input(X):
            raise TypeError("Input data must be a numpy array, pandas DataFrame, list, pandas Series, or Scipy Matrix.")
        
        self.__split_train_val_test(X, y, test_size=test_size, valid_size=valid_size, random_state=random_state)

        self.training_accuracies = []
        self.validation_accuracies = []
        self.epochs = range(1, epochs + 1)

        for epoch in self.epochs:
            if epoch % 5 == 0:
                print(f"{self.model_name} begins epoch {epoch}")
            try:
                self.__fitting_model(self.X_train, self.y_train)
            except TypeError:
                self.X_train = self.X_train.toarray()
                self.__fitting_model(self.X_train, self.y_train)
            y_pred_train = self.model.predict(self.X_train)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            self.training_accuracies.append(train_accuracy)
            if valid_size != 0:
                y_pred_val = self.model.predict(self.X_val)
                val_accuracy = accuracy_score(self.y_val, y_pred_val)
                self.validation_accuracies.append(val_accuracy)

        if save_model:
            try:
                self.__save_model(self.model, f"./models/normal/{self.model_name}_{self.data_name}_normal.joblib")
            except Exception as e:
                print(f"Error saving model: {e}")
                trb.print_exc()

        self.epoch_training = True
        del self.X_train, self.y_train

    def train_with_finding_best_parameters(
            self, 
            X, 
            y, 
            mode="grid", 
            score_priority="accuracy", 
            random_state=42, 
            test_size=0.2, 
            valid_size=0,
            cv=5, 
            n_jobs=5,
            scores:list=["accuracy", "precision", "recall", "f1"], 
            save_model=False):
        if not self.__is_valid_input(X):
            raise TypeError("Input data must be a numpy array, pandas DataFrame, list, or pandas Series.")
        
        self.__split_train_val_test(
            X, 
            y, 
            test_size=test_size, 
            valid_size=valid_size, 
            random_state=random_state)

        if mode.lower() == "grid":
            if self.__get_total_number_of_combination() > 60:
                self.grid = RandomizedSearchCV(
                    self.model,
                    self.model_parameters,
                    cv=cv,
                    n_jobs=n_jobs,
                    n_iter=60,
                    random_state=random_state,
                    scoring=scores,
                    refit=score_priority
                )
            else:
                self.grid = GridSearchCV(
                    self.model, 
                    self.model_parameters,
                    cv=cv,
                    n_jobs=n_jobs, 
                    scoring=scores,
                    refit=score_priority)
        elif mode.lower() == "random":
            ideal = self.__get_reduced_number_of_combination()

            self.grid = RandomizedSearchCV(
                self.model,
                self.model_parameters,
                cv=cv,
                n_jobs=n_jobs,
                n_iter=ideal,
                random_state=random_state,
                scoring=scores,
                refit=score_priority
            )
        else:
            raise ValueError("Unknown mode")
        try:
            self.grid.fit(self.X_train, self.y_train)
        except TypeError:
            self.X_train = self.X_train.toarray()
            self.grid.fit(self.X_train, self.y_train)
        self.grid_searching = True

        if save_model:
            try:
                self.__save_model(self.grid, f"./models/grid/{self.model_name}_{self.data_name}_grid.joblib")
            except Exception as e:
                print(f"Error saving model: {e}")
                trb.print_exc()

    def plot_train_val_accuracy_after_epochs(self, xlabel="X Label", ylabel="Y Label", save_plot=False):
        if not self.__is_trained():
            raise ValueError("Model has not been trained yet.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.epochs, 
            self.training_accuracies, 
            color="blue", 
            linestyle="--", 
            linewidth=2, 
            label="Training Accuracy")
        if self.X_val is not None:
            plt.plot(
                self.epochs, 
                self.validation_accuracies, 
                color="red", 
                linestyle="--", 
                linewidth=2,
                label="Validation Accuracy")
        plt.title(f"Train-Val Accuracy Results for {self.model_name}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        if save_plot:
            plt.savefig(f"./figs/normal/{self.model_name}_{self.data_name}_normal.jpg")
        plt.show()

    def get_best_estimator(self, put_right_in_the_model=False, free_memory=False):
        """
        Get the best estimator from the grid search.
        
        Parameters:
        - put_right_in_the_model: If True, set the best estimator as the model.
        
        Returns:
        - The best estimator from the grid search.
        """
        if not self.grid_searching:
            raise ValueError("Grid search has not been performed yet.")
        
        self.best_estimator = self.grid.best_estimator_
        print(f"The best estimator for {self.model_name} of dataset {self.data_name} is: ")
        print(self.best_estimator)
        
        if put_right_in_the_model:
            self.model = self.best_estimator

        if free_memory:
            del self.X_train, self.y_train

    def evaluate(self, detailed=False):
        """
        Evaluate the model using the test set and return the classification report.
        Raises:
        - EvaluateError: If the model has not been trained yet.
        """
        
        y_pred = self.model.predict(self.X_test)

        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        if detailed:
            self.metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "weighted_precision": precision_score(self.y_test, y_pred, average='weighted'),
                "wighted_recall": recall_score(self.y_test, y_pred, average='weighted'),
                "weighted_f1": f1_score(self.y_test, y_pred, average='weighted'),
                "macro_precision": precision_score(self.y_test, y_pred, average='macro'),
                "macro_recall": recall_score(self.y_test, y_pred, average='macro'),
                "macro_f1": f1_score(self.y_test, y_pred, average='macro'),
                "roc_auc": roc_auc_score(self.y_test, y_pred)
            }
            detailed_metrics = {
                "dataset": self.data_name,
                "model": self.model_name,
                "type": "grid_search" if self.grid_searching else "normal",
                "metrics": self.metrics,
                "confusion_matrix": self.confusion_matrix
            }
            if not self.grid_searching:
                if hasattr(self, "epochs"):
                    detailed_metrics["epochs"] = len(self.epochs)
                return detailed_metrics
            else:
                if hasattr(self, "epochs"):
                    detailed_metrics["epochs"] = len(self.epochs)
                self.grid_searching = False
                detailed_metrics["best_parameters"] = self.grid.best_params_
                detailed_metrics["best_score"] = self.grid.best_score_
                return detailed_metrics

        self.grid_searching = False
        return classification_report(self.y_test, y_pred)

    
    def print_simple_confusion_matrix(self):
        if self.confusion_matrix is None:
            raise ValueError("Model is not evaluated.")
        
        print(f"Confusion Matrix for {self.model_name} on dataset {self.data_name}:")
        print("Pattern:")
        print("True Negative (TN) | False Positive (FP)")
        print("False Negative (FN) | True Positive (TP)")
        print(self.confusion_matrix)
        print("\n")
    
    def draw_confusion_matrix(self):
        if self.confusion_matrix is None:
            raise ValueError("Model is not evaluated.")
        
        disp = ConfusionMatrixDisplay(
            self.confusion_matrix, 
            display_labels=self.model.classes_
        )

        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{self.model_name} Confusion Matrix")
        plt.show()

    def __fitting_model(self, X, y, partial=False):
        if not partial:
            self.model.fit(X, y)
        else:
            if isinstance(self.model, (
                RandomForestClassifier,
                DecisionTreeClassifier,
                AdaBoostClassifier,
                SVC,
                HistGradientBoostingClassifier
            )):
                raise TypeError("The model does not support partial fitting.")
            self.model.partial_fit(X, y)

    def __is_trained(self):
        """
        Check if the model has been trained.
        Returns:
        - True if the model has been trained, False otherwise.
        """
        try:
            check_is_fitted(self.model)
            return True
        except NotFittedError:
            return False
    
    def __is_valid_input(self, X):
        return isinstance(X, (
            np.ndarray, 
            pd.DataFrame, 
            list, 
            pd.Series,
            scipy.sparse._csr.csr_matrix
            ))
    
    def __save_model(self, model, path):
        """
        Save the trained model to a file.
        
        Parameters:
        - filename: The name of the file to save the model.
        """
        joblib.dump(model, path)

    def __get_total_number_of_combination(self):
        self.total_combinations = math.prod(len(v) for v in self.model_parameters.values())

        return self.total_combinations

    def __get_reduced_number_of_combination(self):

        total_combinations = math.prod(len(v) for v in self.model_parameters.values())

        if total_combinations < 10:
            return total_combinations - 1
        elif total_combinations < 30:
            return 20
        elif total_combinations < 60:
            return int(total_combinations / 2) + 10
        elif total_combinations < 80:
            return 45
        else:
            return int(total_combinations / 2) + 20

class TransformerClassificationModel:
    def __init__(self, data_input, model_checkpoint, x_features, y_feature):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.y_feature = y_feature
        if isinstance(data_input, list):
            self.dataset = Dataset.from_list(data_input)
        elif isinstance(data_input, dict):
            self.dataset = Dataset.from_dict(data_input)
        elif isinstance(data_input, pd.DataFrame):
            self.dataset = Dataset.from_pandas(data_input)
        elif isinstance(data_input, str):
            file_extension = os.path.splitext(data_input)[1]
            if file_extension == ".csv":
                df = pd.read_csv(data_input)
                self.dataset = Dataset.from_csv(data_input)
            elif file_extension == ".sql":
                self.dataset = Dataset.from_sql(data_input)
            else:
                raise ValueError("The string input must refer to the file path.")
            
    def __set_label_encoder_decoder(self, features):
        features = sorted(set(features))
        self.__label2id = {label: i for i, label in enumerate(features)}
        self.__id2label = {i: label for label, i in self.__label2id.items()}
            
    def set_model(self, y_feature):
        self.__set_label_encoder_decoder(self.dataset[y_feature])

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            id2label=self.__id2label,
            label2id=self.__label2id
        )

    def train_test_val_split(self, test_size, valid_size=0, random_state=42):
        if (valid_size < 0 or test_size < 0):
            raise ValueError("Ratio must not be a negative number.")
        if (test_size + valid_size) >= 1:
            raise ValueError("Total ratio of testing and validation set must not larger or equal to 1.")

        train_temp = self.dataset.train_test_split(test_size + valid_size, seed=random_state)
        self.__train_set = train_temp["train"]

        if valid_size != 0:
            val_ratio = valid_size / (test_size + valid_size)

            test_temp = train_temp["test"].train_test_split(val_ratio)
            self.__validation_set = test_temp["train"]
            self.__test_set = test_temp["test"]

            self.dataset = DatasetDict({
                "train": self.__train_set,
                "validation": self.__validation_set,
                "test": self.__test_set
            })

        else:
            self.__test_set = train_temp["test"]

            self.dataset = DatasetDict({
                "train": self.__train_set,
                "test": self.__test_set
            })

    def tokenizing(self, input_features:list[str]=None, merged_feature:str=None):
        if input_features is not None and merged_feature is not None:
            merge_preprocess_func = partial(
                self.__merge_features,
                input_features=input_features,
                output_feature=merged_feature
            )

            self.dataset = self.dataset.map(merge_preprocess_func, batched=True)

        preprocess_func = partial(
            self.__preprocess_function,
            feature=merged_feature
        )

        self.dataset = self.dataset.map(preprocess_func)

        self.tokenized_dataset = self.dataset.rename_column(self.y_feature, "label")

        print(self.tokenized_dataset["train"][0])

    def train(self,
              save_path,
              batch_size=8,
              epochs=20,
              metric_monitor="accuracy",
              epoch_limitation=3,
              num_workers=4):
        data_collator = DataCollatorWithPadding(self.tokenizer)

        if not isinstance(self.tokenized_dataset["train"][0]["label"], int):
            self.tokenized_dataset = self.tokenized_dataset.map(self.__encoder)

        print(self.tokenized_dataset)

        self.__training_args = TrainingArguments(
            output_dir=save_path,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=2e-5,
            load_best_model_at_end=True,
            metric_for_best_model=metric_monitor,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            weight_decay=0.01,
            dataloader_num_workers=num_workers,
            remove_unused_columns=True,
            auto_find_batch_size=True
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.__training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            compute_metrics=self.__compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=epoch_limitation)],
            data_collator=data_collator
        )

        self.trainer.train()

    def plot_training_validation_loss(self, xlabel="Epoch", ylabel="Loss"):
        log_history = self.trainer.state.log_history

        train_losses = []
        val_losses = []
        epochs = []

        for log in log_history:
            if "loss" in log and "epoch" in log:
                train_losses.append(log["loss"])
                epochs.append(log["epoch"])
            if "eval_loss" in log and "epoch" in log:
                val_losses.append(log["eval_loss"])

        plt.plot(epochs[:len(train_losses)], train_losses, label='Train Loss')
        plt.plot(epochs[:len(val_losses)], val_losses, label='Validation Loss')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Training and Validation Loss of {self.model_checkpoint}')
        plt.legend()
        plt.show()

    def evaluate(self):
        test_pred = self.trainer.predict(self.tokenized_dataset["test"])

        metrics = self.__compute_metrics(test_pred)

        return metrics

    def __merge_features(
            self, 
            example, 
            input_features:list[str]|tuple[str], 
            output_feature:str):
        merged_features = []

        if len(input_features) == 2:
            for feature1, feature2 in zip(
                example[input_features[0]], 
                example[input_features[1]]
            ):
                merged = f"{feature1} [SEP] {feature2}"
                merged_features.append(merged)
        elif len(input_features) == 3:
            for feature1, feature2, feature3 in zip(
                example[input_features[0]], 
                example[input_features[1]],
                example[input_features[2]]
            ):
                merged = f"{feature1} [SEP] {feature2} [SEP] {feature3}"
                merged_features.append(merged)
        elif len(input_features) == 4:
            for feature1, feature2, feature3, feature4 in zip(
                example[input_features[0]], 
                example[input_features[1]],
                example[input_features[2]],
                example[input_features[3]]
            ):
                merged = f"{feature1} [SEP] {feature2} [SEP] {feature3} [SEP] {feature4}"
                merged_features.append(merged)
        example[output_feature] = merged_features
        return example

    def __preprocess_function(self, examples, feature):
        return self.tokenizer(examples[feature], truncation=True, padding=True)
    
    def __encoder(self, example):
        example["label"] = self.__label2id[example["label"]]
        return example
    
    def __compute_metrics(self, pred:PredictionOutput):
        logits = pred.predictions
        labels = pred.label_ids

        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "weighted_precision": precision_score(labels, preds, average='weighted'),
            "wighted_recall": recall_score(labels, preds, average='weighted'),
            "weighted_f1": f1_score(labels, preds, average='weighted'),
            "macro_precision": precision_score(labels, preds, average='macro'),
            "macro_recall": recall_score(labels, preds, average='macro'),
            "macro_f1": f1_score(labels, preds, average='macro'),
            "roc_auc": roc_auc_score(labels, preds)
        }

def add_to_json_array(filename, new_object, array_key=None, mode="overwrite"):
    if mode != "overwrite":
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = [] if array_key is None else {array_key: []}
        
        if array_key:
            # JSON structure: {"items": [...]}
            if array_key not in data:
                data[array_key] = []
            data[array_key].append(new_object)
        else:
            # JSON structure: [...]
            data.extend(new_object)
    
    with open(filename, 'w') as f:
        json.dump(new_object, f, indent=4) 

def train_and_evaluate_models(
        X, 
        y, 
        dataset_name, 
        metric_results:list, 
        model=None,
        mode="normal",
        epochs=10,
        n_jobs=6,
        cv=4,
        test_size=0.15,
        valid_size=0.15,
        save_plot:bool=False,
        plot_xlabel="Epochs",
        plot_ylabel="Accuracy",
        threading=True,
        max_workers=3
    ):

    def __t_and_e(
            model,
            X,
            y,
            dataset_name,
            metric_results,
            epochs,
            mode,
            n_jobs,
            cv,
            test_size,
            valid_size,
            save_plot,
            plot_xlabel,
            plot_ylabel
    ):
        print(f"Begin {model.__class__.__name__}")
        classification_model = ClassificationModel(model, dataset_name)
        if mode.lower() == "epochs":
            classification_model.train_with_epochs(X, y, epochs=epochs, save_model=True, test_size=test_size, valid_size=valid_size)
            print(f"{model.__class__.__name__} classification report")
            classification_model.plot_train_val_accuracy_after_epochs(xlabel=plot_xlabel, ylabel=plot_ylabel, save_plot=save_plot)
        elif mode.lower() == "best":
            classification_model.train_with_finding_best_parameters(X, y, save_model=True, mode=mode, n_jobs=n_jobs, cv=cv)
            classification_model.get_best_estimator(put_right_in_the_model=True)
            print(f"{model.__class__.__name__} classification report")
        elif mode.lower() == "normal":
            classification_model.train(X, y)
            print(f"{model.__class__.__name__} classification report")
        else:
            raise ValueError("Unknown mode. The avaiable modes are 'epochs', 'best', and 'normal'")
        metrics = classification_model.evaluate(detailed=True)
        metric_results.append(metrics)
        print(metrics)
        print("\n")

    train_eval = partial(
        __t_and_e,
        X=X,
        y=y,
        dataset_name=dataset_name,
        metric_results=metric_results,
        mode=mode,
        epochs=epochs,
        n_jobs=n_jobs,
        cv=cv,
        test_size=test_size,
        valid_size=valid_size,
        save_plot=save_plot,
        plot_xlabel=plot_xlabel,
        plot_ylabel=plot_ylabel
    )

    if model is None:
        models = get_classification_models(X)
        if threading:
            print("Threading avaiable")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(train_eval, models)
        else:
            for m in models:
                train_eval(m)
    else:
        train_eval(model)
import scipy.sparse
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, LinearSVC
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
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import faiss
import matplotlib.pyplot as plt
import numpy as np
import traceback as trb
import pandas as pd
import scipy
from scipy.spatial import distance
import joblib
import json
import math

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

class CustomKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

def get_classification_models(nomlp = True):
    if not nomlp:
        return [
            SVC(),
            MultinomialNB(),
            BernoulliNB(),
            RandomForestClassifier(n_jobs=4),
            DecisionTreeClassifier(),
            AdaBoostClassifier(),
            LogisticRegression(n_jobs=4),
            KNeighborsClassifier(n_jobs=4),
            SGDClassifier(),
            Perceptron(),
            PassiveAggressiveClassifier(n_jobs=4),
            MLPClassifier(
                hidden_layer_sizes=(100, ), 
                early_stopping=True,
                n_iter_no_change=6
            ),
            ExtraTreesClassifier(n_jobs=4),
            XGBClassifier()
        ]
    else:
        return [
            KNeighborsClassifier(n_jobs=5),
            LinearSVC(),
            MultinomialNB(),
            BernoulliNB(),
            RandomForestClassifier(n_jobs=4),
            DecisionTreeClassifier(),
            AdaBoostClassifier(),
            LogisticRegression(n_jobs=4),
            SGDClassifier(),
            Perceptron(),
            PassiveAggressiveClassifier(n_jobs=4),
            ExtraTreesClassifier(n_jobs=4),
            XGBClassifier(),
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
        elif isinstance(self.model, LinearSVC):
            self.model_name = "LinearSVC"
            self.model_parameters = {}
        elif isinstance(self.model, CustomKNeighbors):
            self.model_name = "K-nearest Neighbors"
            self.model_parameters = {}
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

    def __reduce_dim(self, n_dim=100):
        print("Begin dimensional reduction")
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        print(self.X_train)
        print(self.X_test)

        pca = IncrementalPCA(n_components=n_dim, batch_size=1000)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.fit_transform(self.X_test)

        print(self.X_train)
        print(self.X_test)

        print("Finish dimensional reduction")

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
            if isinstance(self.model, KNeighborsClassifier):
                if X.shape[1] > 10000:
                    self.__reduce_dim()
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

    def train_with_epochs(self, X, y, epochs=10, random_state=42, test_size=0.2, valid_size=0, save_model=False, reduce_dim=False):
        if not self.__is_valid_input(X):
            raise TypeError("Input data must be a numpy array, pandas DataFrame, list, pandas Series, or Scipy Matrix.")
        
        self.__split_train_val_test(X, y, test_size=test_size, valid_size=valid_size, random_state=random_state)

        if reduce_dim:
            self.__reduce_dim(100)

        self.training_accuracies = []
        self.validation_accuracies = []
        self.epochs = range(1, epochs + 1)

        if isinstance(self.model, KNeighborsClassifier):
            print("Skipping epochs training due to longevity.")
            self.__fitting_model(self.X_train, self.y_train)

            y_pred_train = self.model.predict(self.X_train)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            self.training_accuracies = [train_accuracy] * len(self.epochs)
            if valid_size != 0:
                y_pred_val = self.model.predict(self.X_val)
                val_accuracy = accuracy_score(self.y_val, y_pred_val)
                self.validation_accuracies = [val_accuracy] * len(self.epochs)
        else:
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

class ClassificationModel2:
    def __init__(self, X, y, dataset_name:str):
        self.X = X
        self.y = y
        self.dataset_name = dataset_name

    def split(self, test_size=0.1, valid_size=0.1, random_state=42):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if not 0 <= valid_size < 1:
            raise ValueError("valid_size must be between 0 and 1")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        if valid_size > 0:
            val_ratio = valid_size / (1 - test_size)

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=val_ratio, random_state=random_state
            )

            print("Val size: ", self.X_test.shape)
        

    def train_with_epochs(self, model:BaseEstimator, epochs, valid_size=0, save_model=False, reduce_dim=False):
        if reduce_dim:
            self.__reduce_dim(100)

        training_accuracies = []
        validation_accuracies = []
        self.epochs = range(1, epochs + 1)

        if isinstance(model, (
            KNeighborsClassifier,
            AdaBoostClassifier,
            XGBClassifier,
            Perceptron,
            PassiveAggressiveClassifier,
            MultinomialNB,
            BernoulliNB,
            LinearSVC,
        )):
            print("Skipping epochs training due to longevity.")
            model.fit(self.X_train, self.y_train)

            y_pred_train = model.predict(self.X_train)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            training_accuracies = [train_accuracy] * len(self.epochs)
            if valid_size != 0:
                y_pred_val = model.predict(self.X_val)
                val_accuracy = accuracy_score(self.y_val, y_pred_val)
                validation_accuracies = [val_accuracy] * len(self.epochs)
        elif isinstance(model, (
                DecisionTreeClassifier,
                RandomForestClassifier,
                ExtraTreesClassifier
            )):
            try:
                model.fit(self.X_train, self.y_train)
            except TypeError:
                self.X_train = self.X_train.toarray()
                model.fit(self.X_train, self.y_train)
            y_pred_train = model.predict(self.X_train)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            training_accuracies = [train_accuracy] * len(self.epochs)
            if valid_size != 0:
                for epoch in self.epochs:
                    y_pred_val = model.predict(self.X_val)
                    val_accuracy = accuracy_score(self.y_val, y_pred_val)
                    validation_accuracies.append(val_accuracy)
        else:
            for epoch in self.epochs:
                if epoch % 5 == 0:
                    print(f"{self.model_name} begins epoch {epoch}")
                try:
                    model.fit(self.X_train, self.y_train)
                except TypeError:
                    self.X_train = self.X_train.toarray()
                    model.fit(self.X_train, self.y_train)
                y_pred_train = model.predict(self.X_train)
                train_accuracy = accuracy_score(self.y_train, y_pred_train)
                training_accuracies.append(train_accuracy)
                if valid_size != 0:
                    y_pred_val = model.predict(self.X_val)
                    val_accuracy = accuracy_score(self.y_val, y_pred_val)
                    validation_accuracies.append(val_accuracy)

        model_name = model.__class__.__name__

        if save_model:
            try:
                self.__save_model(model, f"./models/normal/{model_name}_{self.dataset_name}_normal.joblib")
            except Exception as e:
                print(f"Error saving model: {e}")
                trb.print_exc()

        return training_accuracies, validation_accuracies

    def train_and_evaluate_models(self,
        epochs=10,
        valid_size=0.15,
        save_plot:bool=False,
        plot_xlabel="Epochs",
        plot_ylabel="Accuracy",
        threading=True,
        max_workers=3,
        nomlp=False,
        reduce_dim=False,
    ):
        metric_results = []

        def __t_and_e(model):
            print(f"Begin {model.__class__.__name__}")
            training_accuracies, validation_accuracies = self.train_with_epochs(model, epochs, valid_size=valid_size, reduce_dim=reduce_dim)
            print(f"{model.__class__.__name__} classification report")
            self.plot_train_val_accuracy_after_epochs(
                model, 
                training_accuracies,
                validation_accuracies,
                xlabel=plot_xlabel, 
                ylabel=plot_ylabel, 
                save_plot=save_plot)
            metrics = self.evaluate(model, detailed=True)
            del model
            metric_results.append(metrics)
            print(metrics)
            print("\n")

        models = get_classification_models(nomlp=nomlp)
        if threading:
            print("Threading avaiable")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(__t_and_e, models)
        else:
            for m in models:
                __t_and_e(m)

        return metric_results

    def plot_train_val_accuracy_after_epochs(
            self, 
            model:BaseEstimator,
            training_accuracies:list,
            validation_accuracies:list,
            xlabel="X Label", 
            ylabel="Y Label", 
            save_plot=False):        
        print(self.epochs)
        print(len(training_accuracies))
        print(len(validation_accuracies))
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.epochs, 
            training_accuracies, 
            color="blue", 
            linestyle="--", 
            linewidth=2, 
            label="Training Accuracy")
        if self.X_val is not None:
            plt.plot(
                self.epochs, 
                validation_accuracies, 
                color="red", 
                linestyle="--", 
                linewidth=2,
                label="Validation Accuracy")
            
        model_name = model.__class__.__name__

        plt.title(f"Train-Val Accuracy Results for {model_name}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        if save_plot:
            plt.savefig(f"./figs/normal/{model_name}_{self.dataset_name}_normal.jpg")
        plt.show()

    def evaluate(self, model:BaseEstimator, detailed=False):
        """
        Evaluate the model using the test set and return the classification report.
        Raises:
        - EvaluateError: If the model has not been trained yet.
        """
        
        y_pred = model.predict(self.X_test)

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
                "dataset": self.dataset_name,
                "model": model.__class__.__name__,
                "metrics": self.metrics,
                "confusion_matrix": self.confusion_matrix
            }
            if hasattr(self, "epochs"):
                detailed_metrics["epochs"] = len(self.epochs)

            return detailed_metrics
        return classification_report(self.y_test, y_pred)

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
        max_workers=3,
        nomlp=False,
        reduce_dim=False,
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
            classification_model.train_with_epochs(X, y, epochs=epochs, save_model=True, test_size=test_size, valid_size=valid_size, reduce_dim=reduce_dim)
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
        del classification_model, model
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
        plot_ylabel=plot_ylabel,
    )

    if model is None:
        models = get_classification_models(X, nomlp=nomlp)
        if threading:
            print("Threading avaiable")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(train_eval, models)
        else:
            for m in models:
                train_eval(m)
            
    else:
        train_eval(model)
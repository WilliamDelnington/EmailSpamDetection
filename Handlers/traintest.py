import scipy.sparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from keras.src.models import Sequential
from keras.src.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout, LSTM, Bidirectional, SimpleRNN
from keras.src.layers import TextVectorization
import matplotlib.pyplot as plt
import numpy as np
import traceback as trb
import pandas as pd
import scipy
import joblib

model_parameters = {
    "SVM": {
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
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
        "penalty": ["l2", "l1", "elasticnet", None],
        "C": [0.1, 1, 10],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    },
    "KNeighbors": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    },
    "SGDClassifier": {
        "loss": ["hinge", "log", "squared_hinge", "modified_huber"],
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    },
    "GradientBoosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Perceptron": {
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [1000, 2000, 3000],
        "tol": [1e-3, 1e-4, 1e-5],
    },
}

class PreprocessAndTrainWithCNN:
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
                self.X = data.iloc[:, :-1]
                self.y = data.iloc[:, -1]
            elif input_features is None:
                self.X = data[input_features]
                self.y = data[[f for f in data.columns if f not in input_features]]
            elif output_feature is None:
                self.X = data.drop(columns=[output_feature])
                self.y = data[output_feature]
            else:
                self.X = data[input_features]
                self.y = data[output_feature]
            
            if len(self.X.columns) > 1:
                self.X = self.X.astype(str).agg(" ".join, axis=1)
            self.X = self.X.to_list()
            self.y = np.array(self.y, dtype=int)
            print(self.X)
            print(self.y) 

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

        self.x_train = np.array(self.x_train, dtype=object)
        self.x_test = np.array(self.x_test, dtype=object)

    def preprocess(self, max_features=10000, sequence_length=100):
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
              pooling_dropout=False, 
              pooling_dropout_rate=0.2,
              dense_dropout=False,
              dense_dropout_rate=0.5,
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
            activation='relu'
        ))

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

        self.model.fit(
            self.x_train, 
            self.y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(self.x_test, self.y_test)
        )

        self.model.save("./Classify_CNN_model.h5")
    
    def evaluate(self):
        if not hasattr(self, 'x_test') or not hasattr(self, 'y_test'):
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = (y_pred > 0.5).astype(int)

        report = classification_report(self.y_test, y_pred_classes)
        print(report)
    

class PreprocessAndTrainWithRNN:
    def __init__(self):
        self.model = Sequential()

    def load_data(self, 
                  data,
                  input_features:list=None,
                  output_feature:str=None
                  ):
        if isinstance(data, pd.DataFrame):
            if input_features is None and output_feature is None:
                print("The input features are not specified. The last column will be used as the output feature, while the rest will be used as input features.")
                self.X = data.iloc[:, :-1]
                self.y = data.iloc[:, -1]
            elif input_features is None:
                self.X = data[input_features]
                self.y = data[[f for f in data.columns if f not in input_features]]
            elif output_feature is None:
                self.X = data.drop(columns=[output_feature])
                self.y = data[output_feature]
            else:
                self.X = data[input_features]
                self.y = data[output_feature]
            
            if len(self.X.columns) > 1:
                self.X = self.X.astype(str).agg(" ".join, axis=1)
            self.X = self.X.to_list()
            self.y = np.array(self.y, dtype=int)
            print(self.X)
            print(self.y) 

        elif isinstance(data, (np.ndarray, list)):
            self.X = data[:, :-1]
            self.y = data[:, -1]

        else:
            raise TypeError("Unsupported data type. Please provide a DataFrame, numpy array, or list.")
        
    def split(self,
              test_size=0.2, 
              random_state=42):
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            raise ValueError("Data has not been loaded. Call load_data() first.")
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        self.x_train = np.array(self.x_train, dtype=object)
        self.x_test = np.array(self.x_test, dtype=object)

    def preprocess(self, max_features=10000, sequence_length=100):
        self.vectorizer = TextVectorization(
            max_tokens=max_features, 
            output_mode='int', 
            output_sequence_length=sequence_length
        )
        self.vectorizer.adapt(self.x_train)

    def build(self, max_features=10000, 
              embedding_size=128, 
              units=64,
              hidden_size=128,
              pooling_dropout=False, 
              pooling_dropout_rate=0.2,
              dense_dropout=False,
              dense_dropout_rate=0.5,
              recurrent_method="simple",
              bidirectional=False,
              epochs=10,
              batch_size=32):
        
        self.model.add(self.vectorizer)
    
        self.model.add(Embedding(
            input_dim=max_features, 
            output_dim=embedding_size
        ))

        if recurrent_method == "simple":
            layer = SimpleRNN(units, return_sequences=True)

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

        self.model.fit(
            self.x_train, 
            self.y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(self.x_test, self.y_test)
        )

        self.model.save("./Classify_CNN_model.h5")
    
    def evaluate(self):
        if not hasattr(self, 'x_test') or not hasattr(self, 'y_test'):
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = (y_pred > 0.5).astype(int)

        report = classification_report(self.y_test, y_pred_classes)
        print(report)

models = [
    SVC(),
    MultinomialNB(),
    BernoulliNB(),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    LogisticRegression(),
    KNeighborsClassifier(),
    SGDClassifier(),
    GradientBoostingClassifier(),
    Perceptron(),
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
        self.grid_searching = False if not self.__is_trained() else True
        self.normal_training = False if not self.__is_trained() else True

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
            self.__fitting_model(self.X_train, self.y_train, partial=partial)
        self.normal_training = True

        if save_model:
            try:
                self.__save_model(self.model, f"./models/{self.model_name}_{self.data_name}_normal.joblib")
            except Exception as e:
                print(f"Error saving model: {e}")
                trb.print_exc()

    # def train_with_epochs(self, X, y, epochs=10, batch_size=32, random_state=42, test_size=0.2, valid_size=0):
    #     if not self.__is_valid_input(X):
    #         raise TypeError("Input data must be a numpy array, pandas DataFrame, list, pandas Series, or Scipy Matrix.")
        
    #     self.__split_train_val_test(X, y, test_size=test_size, valid_size=valid_size, random_state=random_state)

    #     if hasattr(self.model, 'fit'):
    #         self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)
    #     else:
    #         raise TypeError("The model does not support training with epochs.")
        
    #     self.normal_training = True

    def validation(self, X, y, score_priority="accuracy", random_state=42, test_size=0.2, valid_size=0, save_model=False):
        if not self.__is_valid_input(X):
            raise TypeError("Input data must be a numpy array, pandas DataFrame, list, or pandas Series.")
        
        self.__split_train_val_test(X, y, test_size=test_size, valid_size=valid_size, random_state=random_state)

        self.grid = GridSearchCV(
            self.model, 
            self.model_parameters, 
            cv=5, 
            scoring=[
                "accuracy", 
                "precision", 
                "recall", 
                "f1"
            ],
            refit=score_priority)
        self.grid.fit(self.X_train, self.y_train)
        self.grid_searching = True

        if save_model:
            try:
                self.__save_model(self.grid, f"./models/{self.model_name}_{self.data_name}_grid.joblib")
            except Exception as e:
                print(f"Error saving model: {e}")
                trb.print_exc()

    def plot_grid(self, x_features, y_features, xlabel="X Label", ylabel="Y Label"):
        if not self.grid_searching:
            raise ValueError("Grid search has not been performed yet.")
        
        results = pd.DataFrame(self.grid.cv_results_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            results[x_features], 
            results[y_features], 
            marker='o', 
            linestyle='-', 
            color='b'
        )
        plt.title(f"Grid Search Results for {self.model_name}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.show()


    def get_best_estimator(self, put_right_in_the_model=False):
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

    def evaluate(self, detailed=False):
        """
        Evaluate the model using the test set and return the classification report.
        Raises:
        - EvaluateError: If the model has not been trained yet.
        """
        if not self.__is_trained():
            raise EvaluateError("The model hasn't been trained yet.")
        
        y_pred = self.model.predict(self.X_test)

        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        if detailed:
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "weighted_precision": precision_score(self.y_test, y_pred, average='weighted'),
                "wighted_recall": recall_score(self.y_test, y_pred, average='weighted'),
                "weighted_f1": f1_score(self.y_test, y_pred, average='weighted'),
                "macro_precision": precision_score(self.y_test, y_pred, average='macro'),
                "macro_recall": recall_score(self.y_test, y_pred, average='macro'),
                "macro_f1": f1_score(self.y_test, y_pred, average='macro'),
                "roc_auc": roc_auc_score(self.y_test, y_pred)
            }
            return metrics

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
                GradientBoostingClassifier
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
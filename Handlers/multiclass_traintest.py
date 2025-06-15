from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
)
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    
)
from sklearn.multiclass import OneVsRestClassifier
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import joblib
import traceback as trb
import matplotlib.pyplot as plt


def get_classification_models():
    return [
        LogisticRegression(multi_class="multinomial"),
        SGDClassifier(n_jobs=2),
        Perceptron(n_jobs=2),
        LinearSVC(),
        PassiveAggressiveClassifier(n_jobs=2),
        RandomForestClassifier(n_jobs=5),
        ExtraTreesClassifier(n_jobs=5),
        DecisionTreeClassifier(),
        BernoulliNB()
    ]

class MulticlassificationModel:
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
        self.ori_model = model
        self.data_name = data_name

        if isinstance(self.ori_model, SGDClassifier):
            self.model_name = "Stochastic Gradient Descent"
            self.model = OneVsRestClassifier(self.ori_model, n_jobs=3)
        elif isinstance(self.ori_model, LinearSVC):
            self.model_name = "LinearSVC"
            self.model = OneVsRestClassifier(self.ori_model, n_jobs=3)
        elif isinstance(self.ori_model, Perceptron):
            self.model_name = "Perceptron"
            self.model = OneVsRestClassifier(self.ori_model, n_jobs=3)
        elif isinstance(self.ori_model, PassiveAggressiveClassifier):
            self.model = OneVsRestClassifier(self.ori_model, n_jobs=3)
            self.model_name = "Passive-Aggressive"
        elif isinstance(self.ori_model, LogisticRegression):
            self.model_name = "Logistic Regression"
            self.model = self.ori_model
        elif isinstance(self.ori_model, ExtraTreesClassifier):
            self.model_name = "Extra Trees Classifier"
            self.model = self.ori_model
        elif isinstance(self.ori_model, RandomForestClassifier):
            self.model_name = "Random Forest"
            self.model = self.ori_model
        elif isinstance(self.ori_model, DecisionTreeClassifier):
            self.model_name = "Decision Tree"
            self.model = self.ori_model
        elif isinstance(self.ori_model, BernoulliNB):
            self.model_name = "Bernoulli Naive Bayes"
            self.model = self.ori_model


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

    def train_with_epochs(self, X, y, epochs=10, random_state=42, test_size=0.2, valid_size=0, save_model=False, reduce_dim=False):
        self.__split_train_val_test(X, y, test_size=test_size, valid_size=valid_size, random_state=random_state)

        self.training_accuracies = []
        self.validation_accuracies = []
        self.epochs = range(1, epochs + 1)

        if isinstance(self.ori_model, (Perceptron, LinearSVC, LogisticRegression, BernoulliNB)):
            self.__normal_train(valid_size=valid_size, save_model=save_model)
        else:
            self.__train_with_epochs(test_train_accuracy=True, valid_size=valid_size, save_model=save_model)

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
                "macro_f1": f1_score(self.y_test, y_pred, average='macro')
            }
            detailed_metrics = {
                "dataset": self.data_name,
                "model": self.model_name,
                "metrics": self.metrics,
                "confusion_matrix": self.confusion_matrix
            }
            if hasattr(self, "epochs"):
                detailed_metrics["epochs"] = len(self.epochs)
            return detailed_metrics

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

    def __save_model(self, model, path):
        """
        Save the trained model to a file.
        
        Parameters:
        - filename: The name of the file to save the model.
        """
        joblib.dump(model, path)

    def __train_with_epochs(self, test_train_accuracy=False, valid_size=0, save_model=True):
        for epoch in self.epochs:
            if epoch % 5 == 0 or epoch % 3 == 0:
                print(f"{self.model_name} begins epoch {epoch}")
            self.model.fit(self.X_train, self.y_train)
            if test_train_accuracy:
                y_pred_train = self.model.predict(self.X_train)
                train_accuracy = accuracy_score(self.y_train, y_pred_train)
                self.training_accuracies.append(train_accuracy)
            if valid_size != 0:
                y_pred_val = self.model.predict(self.X_val)
                val_accuracy = accuracy_score(self.y_val, y_pred_val)
                self.validation_accuracies.append(val_accuracy)

        if save_model:
            try:
                self.__save_model(self.grid, f"./models/{self.model_name}_{self.data_name}.joblib")
            except Exception as e:
                print(f"Error saving model: {e}")
                trb.print_exc()

    def __normal_train(self, valid_size=0, save_model=True):
        self.model.fit(self.X_train, self.y_train)
        y_pred_train = self.model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        self.training_accuracies = [train_accuracy] * len(self.epochs)
        if valid_size != 0:
            y_pred_val = self.model.predict(self.X_val)
            val_accuracy = accuracy_score(self.y_val, y_pred_val)
            self.validation_accuracies = [val_accuracy] * len(self.epochs)

    def plot_train_val_accuracy_after_epochs(self, xlabel="X Label", ylabel="Y Label", save_plot=False):
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
            plt.savefig(f"./figs/{self.model_name}_{self.data_name}_normal.jpg")
        plt.show()

def train_and_evaluate_models(
        X, 
        y, 
        dataset_name, 
        metric_results:list, 
        model=None,
        epochs=10,
        test_size=0.15,
        valid_size=0.15,
        save_plot:bool=False,
        plot_xlabel="Epochs",
        plot_ylabel="Accuracy",
        threading=True,
        max_workers=3,
        reduce_dim=False
    ):

    def __t_and_e(
            model,
            X,
            y,
            dataset_name,
            metric_results,
            epochs,
            test_size,
            valid_size,
            save_plot,
            plot_xlabel,
            plot_ylabel
    ):
        print(f"Begin {model.__class__.__name__}")
        classification_model = MulticlassificationModel(model, dataset_name)
        classification_model.train_with_epochs(X, y, epochs=epochs, save_model=True, test_size=test_size, valid_size=valid_size, reduce_dim=reduce_dim)
        print(f"{model.__class__.__name__} classification report")
        classification_model.plot_train_val_accuracy_after_epochs(xlabel=plot_xlabel, ylabel=plot_ylabel, save_plot=save_plot)
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
        epochs=epochs,
        test_size=test_size,
        valid_size=valid_size,
        save_plot=save_plot,
        plot_xlabel=plot_xlabel,
        plot_ylabel=plot_ylabel,
    )

    if model is None:
        models = get_classification_models()
        if threading:
            print("Threading avaiable")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(train_eval, models)
        else:
            for m in models:
                train_eval(m)
            
    else:
        train_eval(model)
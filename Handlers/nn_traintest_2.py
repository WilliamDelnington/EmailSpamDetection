import pandas as pd
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
    MaxPooling1D,
    BatchNormalization,
    Attention,
    Input
)
from keras.src.layers import TextVectorization
from keras.src.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau
)
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np
import traceback as trb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score
)
import os

# Set random seeds for reproducibility
class NeuralNetworkClassifier:
    def __init__(self, 
                 data_name:str,  
                 max_features:int=5000, 
                 input_length:int=200,
                 multi_class:str="binary",
                 num_classes:int=2):
        self.data_name = data_name
        self.max_features = max_features
        self.input_length = input_length
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.models = {}

    def load_data(self, X, y, convert=False, reshaping=False, scaler=False):
        """
        Load and preprocess the data.
        Parameters:
        - X: The input samples and features to be processed.
        - y: The input label from each sample of the dataset
        """
        self.X = X
        self.y = y
        if convert:
            if isinstance(self.X, (pd.DataFrame, pd.Series)):
                self.X = self.X.astype(str).values
            if isinstance(self.y, (pd.DataFrame, pd.Series)):
                self.y = self.y.astype(np.int32).values

        if scaler:
            self.X = self.__scaling(self.X)
        
        if reshaping:
            if isinstance(self.X, (pd.DataFrame, pd.Series)):
                self.X = self.X.to_numpy(dtype=np.float32).reshape((self.X.shape[0], self.X.shape[1], 1))
                

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

    def vectorizing(self, split_method="whitespace", standardize="lower_and_strip_punctuation"):
        self.vectorizer = TextVectorization(
            max_tokens=self.max_features,
            output_mode="int",
            split=split_method,
            standardize=standardize,
            output_sequence_length=self.input_length
        )

        self.vectorizer.adapt(self.X_train)

    def build_CNN(self, embed_and_vectorize=True):
        model = Sequential()
        if embed_and_vectorize:
            model.add(self.vectorizer)
            model.add(
                Embedding(
                    input_dim=self.max_features, 
                    output_dim=128, input_length=self.input_length))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        if self.multi_class == "binary":
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(
            loss='binary_crossentropy' if self.multi_class=="binary" else "sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=[
                'accuracy', 
                # 'precision', 
                # 'recall'
            ]
        )

        model.summary()

        return model
    
    def build_RNN(self, embed_and_vectorize=True, bidirectional=False):
        model = Sequential()
        if embed_and_vectorize:
            model.add(self.vectorizer)
            model.add(
                Embedding(
                    input_dim=self.max_features, 
                    output_dim=128, input_length=self.input_length))
        if bidirectional:
            model.add(Bidirectional(GRU(128, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Bidirectional(GRU(32)))
            model.add(Dropout(0.3))
        else:
            model.add(GRU(128, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(GRU(32))
            model.add(Dropout(0.3))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        if self.multi_class == "binary":
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(
            loss='binary_crossentropy' if self.multi_class=="binary" else "sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=[
                'accuracy', 
                # 'precision', 
                # 'recall'
            ]
        )

        model.summary()

        return model

    def build_CNN_LSTM(self, embed_and_vectorize=True, bidirectional=False):
        model = Sequential()
        if embed_and_vectorize:
            model.add(self.vectorizer)
            model.add(
                Embedding(
                    input_dim=self.max_features, 
                    output_dim=128, input_length=self.input_length))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))

        if bidirectional:
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(32)))
            model.add(Dropout(0.3))
        else:
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(32))
            model.add(Dropout(0.3))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        if self.multi_class == "binary":
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(
            loss='binary_crossentropy' if self.multi_class=="binary" else "sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=[
                'accuracy', 
                # 'precision', 
                # 'recall'
            ]
        )

        model.summary()

        return model
    
    def build_ANN(self, embed_and_vectorize=True):
        model = Sequential()
        if embed_and_vectorize:
            model.add(self.vectorizer)
            model.add(
                Embedding(
                    input_dim=self.max_features, 
                    output_dim=128, 
                    input_length=self.input_length
                ))

        model.add(GlobalMaxPooling1D())

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))

        if self.multi_class == "binary":
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(
            loss='binary_crossentropy' if self.multi_class=="binary" else "sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=[
                'accuracy', 
                # 'precision', 
                # 'recall'
            ]
        )

        model.summary()

        return model

    def train_model(self, model, model_name, monitor='val_loss', epochs=20, patience=6, verbose=1, batch_size=32, save_model=True):
        callback = [EarlyStopping(
            monitor=monitor,
            patience=patience,
            verbose=verbose,
            min_delta=0.001,
            restore_best_weights=True
        )]

        history = model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=callback,
            validation_data=(self.X_val, self.y_val)
        )

        stopped_epoch = callback[0].stopped_epoch

        epochs = stopped_epoch + 1 if stopped_epoch != 0 else epochs

        if save_model:
            model.save(f"./models/Classify_{self.data_name}_{model_name}_model.h5")

        self.models[model_name] = {
            "model": model,
            "history": history,
            "epochs": epochs
        }

        return history

    def plot_training_validation_accuracy(
            self,
            model_name,
            figsize=(10, 6),
            plot_xlabel="Epochs",
            plot_ylabel="Accuracy",
            save_plot=True,
            parent_folder="./figs/"):
        history = self.models[model_name]["history"]
        epochs = self.models[model_name]["epochs"]
        accuracy = history.history["accuracy"]
        plt.figure(figsize=figsize)
        plt.plot(
            range(1, epochs + 1),
            accuracy,
            color="blue", 
            linestyle="--", 
            linewidth=2, 
            label="Training Accuracy"
        )
        if hasattr(self, "X_val"):
            val_accuracy = history.history["val_accuracy"]
            plt.plot(
                range(1, epochs + 1),
                val_accuracy,
                color="red", 
                linestyle="--", 
                linewidth=2, 
                label="Validation Accuracy"
            )
        plt.title(f"{model_name} Train-Val Accuracy Results for {self.data_name}")
        plt.xlabel(xlabel=plot_xlabel)
        plt.ylabel(ylabel=plot_ylabel)
        plt.grid(True, alpha=0.3)
        if save_plot:
            plt.savefig(os.path.join(parent_folder, f"{model_name}_{self.data_name}.jpg"))
        plt.show()

    def plot_training_validation_loss(
            self,
            model_name,
            figsize=(10, 6),
            plot_xlabel="Epochs",
            plot_ylabel="Loss",
            save_plot=True,
            parent_folder="./figs/"):
        history = self.models[model_name]["history"]
        epochs = self.models[model_name]["epochs"]
        losses = history.history["loss"]
        plt.figure(figsize=figsize)
        plt.plot(
            range(1, epochs + 1),
            losses,
            color="green", 
            linestyle="--", 
            linewidth=2, 
            label="Training Loss"
        )
        if hasattr(self, "X_val"):
            val_losses = history.history["val_loss"]
            plt.plot(
                range(1, epochs + 1),
                val_losses,
                color="crimson", 
                linestyle="--", 
                linewidth=2, 
                label="Validation Loss"
            )
        plt.title(f"{model_name} Train-Val Loss Results for {self.data_name}")
        plt.xlabel(xlabel=plot_xlabel)
        plt.ylabel(ylabel=plot_ylabel)
        plt.grid(True, alpha=0.3)
        if save_plot:
            plt.savefig(os.path.join(parent_folder, f"{model_name}_{self.data_name}_loss.jpg"))
        plt.show()

    def evaluate(self, model_name, detailed=True):
        model = self.models[model_name]["model"]
        epochs = self.models[model_name]["epochs"]
        y_pred = model.predict(self.X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)

        cm = confusion_matrix(self.y_test, y_pred_classes)
        if detailed:
            metrics = {
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
                "model": model_name,
                "metrics": metrics,
                "confusion_matrix": cm,
                "epochs": epochs
            }
            return detailed_metrics

        return classification_report(self.y_test, y_pred)
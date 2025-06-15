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

class NeuralNetworkClassifier(ABC):
    def __init__(self, 
                 data_name:str, 
                 model_name:str, 
                 max_features:int, 
                 input_length:int,
                 multi_class:str,
                 num_classes:int):
        self.model = Sequential()
        self.epochs = 10
        self.history = None
        self.data_name = data_name
        self.model_name = model_name
        self.max_features = max_features
        self.input_length = input_length
        self.multi_class = multi_class
        self.num_classes = num_classes

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

    def reduce_dim(self, input, num_features=100):
        try:
            l = self.X.shape
            print(l)
        except Exception as e:
            print(f"An error when getting the shape: {trb.print_exc()}")
            l = len(self.X[0])
        svd = TruncatedSVD(n_components=num_features)
        self.X_train = svd.fit_transform(input)
        
    def vectorizing(self, data_input=None, print_shape=True, split_method="whitespace", standardize="lower_and_strip_punctuation"):
        self.vectorizer = TextVectorization(
            max_tokens=self.max_features,
            output_mode="int",
            split=split_method,
            standardize=standardize,
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

    def plot_training_validation_loss(
            self,
            figsize=(10, 6),
            plot_xlabel="Epochs",
            plot_ylabel="Loss",
            save_plot=True,
            parent_folder="./figs/"):
        if self.history is None:
            raise ValueError("Model is not trained. Call build() method before plotting.")
        losses = self.history.history["loss"]
        plt.figure(figsize=figsize)
        plt.plot(
            range(1, self.epochs + 1),
            losses,
            color="green", 
            linestyle="--", 
            linewidth=2, 
            label="Training Loss"
        )
        if hasattr(self, "X_val"):
            val_losses = self.history.history["val_loss"]
            plt.plot(
                range(1, self.epochs + 1),
                val_losses,
                color="crimson", 
                linestyle="--", 
                linewidth=2, 
                label="Validation Loss"
            )
        plt.title(f"{self.model_name} Train-Val Loss Results for {self.data_name}")
        plt.xlabel(xlabel=plot_xlabel)
        plt.ylabel(ylabel=plot_ylabel)
        plt.grid(True, alpha=0.3)
        if save_plot:
            plt.savefig(os.path.join(parent_folder, f"{self.model_name}_{self.data_name}_loss.jpg"))
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
    
    def __scaling(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

class ConvolutionalNNClassifier(NeuralNetworkClassifier):
    # def __init__(self, vocab_size, embedding_size, num_filters, filter_sizes, hidden_size, output_size, dropout=0.2):
    def __init__(self, data_name, model_name="CNN", max_features=5000, input_length=200, multi_class="binary", num_classes=4):
        super().__init__(data_name, model_name, max_features, input_length, multi_class, num_classes)

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

    def complex_build(self,
                      callback_methods=["early stopping"],
                      epoch_limitation=7,
                      epochs=20,
                      batch_size=32,
                      embed_and_vectorize=True,
                      save_model=True,
                      input_shape=(641, 1)):
        if embed_and_vectorize:
            self.model.add(self.vectorizer)
            self.model.add(
                Embedding(
                    input_dim=self.max_features, 
                    output_dim=128, input_length=self.input_length))
        if input_shape:
            self.model.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape))
        else:
            self.model.add(Conv1D(128, kernel_size=3, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv1D(128, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.3))

        self.model.add(Conv1D(64, kernel_size=3, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv1D(64, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.3))

        self.model.add(Conv1D(32, kernel_size=3, activation='relu'))
        self.model.add(GlobalMaxPooling1D())

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.4))
        if self.multi_class == "binary":
            self.model.add(Dense(1, activation='sigmoid'))
        else:
            self.model.add(Dense(self.num_classes, activation="softmax"))

        self.model.compile(
            loss='binary_crossentropy' if self.multi_class=="binary" else "sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=[
                'accuracy', 
                # 'precision', 
                # 'recall'
            ]
        )

        self.model.summary()

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

class RecurrentNNClassifier(NeuralNetworkClassifier):
    def __init__(self, data_name, model_name="RNN", max_features=5000, input_length=200, multi_class="binary", num_classes=4):
        super().__init__(data_name, model_name, max_features, input_length, multi_class, num_classes)

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
            loss='binary_crossentropy' if self.multi_class=="binary" else "sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=[
                'accuracy', 
                # 'precision', 
                # 'recall'
            ]
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

    def complex_build(self,
                      callback_methods=["early stopping"],
                      epoch_limitation=7,
                      epochs=20,
                      batch_size=32,
                      embed_and_vectorize=True,
                      save_model=True):
        if embed_and_vectorize:
            self.model.add(self.vectorizer)
            self.model.add(Embedding(
                    input_dim=self.max_features,
                    output_dim=128, 
                    input_length=self.input_length))

        self.model.add(Bidirectional(GRU(128, return_sequences=True)))
        self.model.add(Dropout(0.3))

        self.model.add(Bidirectional(GRU(64, return_sequences=False)))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(64, activation='relu'))

        if self.multi_class == "binary":
            self.model.add(Dense(1, activation='sigmoid'))
        else:
            self.model.add(Dense(self.num_classes, activation="softmax"))

        self.model.compile(
            loss='binary_crossentropy' if self.multi_class=="binary" else "sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=[
                'accuracy', 
                # 'precision', 
                # 'recall'
            ]
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

class ArtificialNNClassifier(NeuralNetworkClassifier):
    def __init__(self, data_name, model_name="ANN", max_features=5000, input_length=200, multi_class="binary", num_classes=4):
        super().__init__(data_name, model_name, max_features, input_length, multi_class, num_classes)

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
            loss='binary_crossentropy' if self.multi_class=="binary" else "sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=[
                'accuracy', 
                # 'precision', 
                # 'recall'
            ]
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

    def complex_build(self,
                    callback_methods=["early stopping"],
                    embed_and_vectorize=True,
                      epoch_limitation=7,
                      epochs=20,
                      batch_size=32,
                      save_model=True):
        if embed_and_vectorize:
            self.model.add(self.vectorizer)
            self.model.add(
                Embedding(
                    input_dim=self.max_features, 
                    output_dim=128, 
                    input_length=self.input_length
                ))

        self.model.add(GlobalMaxPooling1D())

        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.3))

        if self.multi_class == "binary":
            self.model.add(Dense(1, activation='sigmoid'))
        else:
            self.model.add(Dense(self.num_classes, activation="softmax"))

        self.model.compile(
            loss='binary_crossentropy' if self.multi_class=="binary" else "sparse_categorical_crossentropy",
            optimizer='adam',
            metrics=[
                'accuracy', 
                # 'precision', 
                # 'recall'
            ]
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
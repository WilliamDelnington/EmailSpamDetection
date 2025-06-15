from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset, Features, Value, ClassLabel
import os
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score
)
import numpy as np
from transformers.trainer_utils import PredictionOutput

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
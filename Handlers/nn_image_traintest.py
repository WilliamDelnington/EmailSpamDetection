import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.src import layers
from keras.src.models import Sequential
from keras.src.utils.image_utils import load_img, img_to_array
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from keras.src.applications.resnet import ResNet50
from keras.src.applications.mobilenet_v3 import MobileNetV3Small
from keras.src.applications.vgg16 import VGG16
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ImageClassificationPipeline:
    def __init__(self, 
        spam_path, 
        natural_path, 
        img_size=(224, 224), 
        batch_size=32
    ):
        self.spam_path = spam_path
        self.natural_path = natural_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.models = {}
        
    def load_and_preprocess_data(self, 
            save=True,
            X_save_path="./data/data_X.npy",
            y_save_path="./data/data_y.npy",
            img_size=(224, 224)
        ):
        """Load and preprocess images from directories"""
        print("Loading and preprocessing data...")
        
        data = []
        
        # Load spam images (label 1)
        for filename in os.listdir(self.spam_path):
            path = os.path.join(self.spam_path, filename)
            try:
                img = Image.open(path).convert('RGB').resize(img_size)
                data.append((np.array(img), 1))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        # Load natural images (label 0)
        for filename in os.listdir(self.natural_path):
            path = os.path.join(self.natural_path, filename)
            try:
                img = Image.open(path).convert('RGB').resize(img_size)
                data.append((np.array(img), 0))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        X, y = zip(*data)

        if save:
            print("Saving np data")
            np.save(X_save_path, X)
            np.save(y_save_path, y)
        
        print(f"Loaded {len(X)} images")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.1, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print("Splitting data...")
        
        # First split: 80% train, 20% temp
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size + val_size, random_state=42, stratify=y
        )

        if val_size != 0:
            val_ratio = val_size / (test_size + val_size)
            
            # Second split: 10% val, 10% test from the 20% temp
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                self.X_test, self.y_test, test_size=val_ratio, random_state=42, stratify=self.y_test
            )

            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def visualize_data_distribution(self):
        """Visualize data distribution and sample images"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        labels_count = pd.Series(self.y_train).value_counts()
        axes[0, 0].bar(['Natural (0)', 'Spam (1)'], [labels_count[0], labels_count[1]], 
                      color=['skyblue', 'salmon'])
        axes[0, 0].set_title('Training Set Class Distribution')
        axes[0, 0].set_ylabel('Number of Images')
        
        # Sample images from each class
        spam_indices = np.where(self.y_train == 1)[0]
        natural_indices = np.where(self.y_train == 0)[0]
        
        # Show sample spam image
        sample_spam_idx = np.random.choice(spam_indices)
        axes[0, 1].imshow(self.X_train[sample_spam_idx])
        axes[0, 1].set_title('Sample Spam Image')
        axes[0, 1].axis('off')
        
        # Show sample natural image
        sample_natural_idx = np.random.choice(natural_indices)
        axes[1, 0].imshow(self.X_train[sample_natural_idx])
        axes[1, 0].set_title('Sample Natural Image')
        axes[1, 0].axis('off')
        
        # Dataset split visualization
        split_data = {
            'Train': len(self.X_train),
            'Validation': len(self.X_val),
            'Test': len(self.X_test)
        }
        axes[1, 1].pie(split_data.values(), labels=split_data.keys(), autopct='%1.1f%%', 
                      colors=['lightblue', 'lightgreen', 'lightcoral'])
        axes[1, 1].set_title('Dataset Split Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualization: pixel intensity distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        spam_pixels = self.X_train[self.y_train == 1].flatten()
        natural_pixels = self.X_train[self.y_train == 0].flatten()
        
        plt.hist(spam_pixels, bins=50, alpha=0.7, label='Spam', color='salmon', density=True)
        plt.hist(natural_pixels, bins=50, alpha=0.7, label='Natural', color='skyblue', density=True)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Density')
        plt.title('Pixel Intensity Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # Mean pixel intensity per image
        spam_mean_intensities = np.mean(self.X_train[self.y_train == 1], axis=(1, 2, 3))
        natural_mean_intensities = np.mean(self.X_train[self.y_train == 0], axis=(1, 2, 3))
        
        plt.hist(spam_mean_intensities, bins=30, alpha=0.7, label='Spam', color='salmon', density=True)
        plt.hist(natural_mean_intensities, bins=30, alpha=0.7, label='Natural', color='skyblue', density=True)
        plt.xlabel('Mean Image Intensity')
        plt.ylabel('Density')
        plt.title('Mean Image Intensity Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def build_resnet50(self):
        """Build ResNet50 model"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        
        return model
    
    def build_mobilenet(self):
        """Build MobileNetV2 model"""
        base_model = MobileNetV3Small(
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        
        return model
    
    def build_vgg16(self):
        """Build VGG16 model"""
        base_model = VGG16(
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=1e-04),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        
        
        return model
    
    def train_model(self, 
                    model, 
                    model_name, 
                    epochs=20,
                    epoch_limitation=6):
        """Train a model"""
        print(f"\nTraining {model_name}...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=epoch_limitation,
            restore_best_weights=True,
            min_delta=1e-04
        )
        
        # Create data generators
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Store model and history
        self.models[model_name] = {
            'model': model,
            'history': history
        }
        
        return history
    
    def evaluate_model(self, model, model_name):
        """Evaluate model and return metrics"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred_proba = model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Natural', 'Spam']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Natural', 'Spam'], 
                    yticklabels=['Natural', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'weighted_precision': precision_score(self.y_test, y_pred, average="weighted"),
            'weighted_recall': recall_score(self.y_test, y_pred, average="weighted"),
            'weighted_f1': f1_score(self.y_test, y_pred, average="weighted"),
            'macro_precision': precision_score(self.y_test, y_pred, average="macro"),
            'macro_recall': recall_score(self.y_test, y_pred, average="macro"),
            'macro_f1': f1_score(self.y_test, y_pred, average="macro"),
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, model_name):
        """Plot training history"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        history = self.models[model_name]['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot training & validation loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} - Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
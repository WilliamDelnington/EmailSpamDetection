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

            print("Train:", np.unique(self.y_train, return_counts=True))
            print("Val:", np.unique(self.y_val, return_counts=True))
            print("Test:", np.unique(self.y_test, return_counts=True))

            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

        print("Train:", np.unique(self.y_train, return_counts=True))
        print("Test:", np.unique(self.y_test, return_counts=True))
        
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

    def build_cnn(self):
        model = Sequential(
            
        )
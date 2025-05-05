import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from functools import lru_cache

@lru_cache(maxsize=1)
def load_processed_data(landmarks_path: str, labels_path: str):
    """
    Loads preprocessed landmarks and labels from files, with caching for efficiency.
    
    :param landmarks_path: Path to landmarks file (CSV or NPY).
    :param labels_path: Path to labels file (NPY).
    :return: Tuple (landmarks, labels).
    """
    if landmarks_path.endswith('.csv'):
        landmarks = pd.read_csv(landmarks_path).values
    elif landmarks_path.endswith('.npy'):
        landmarks = np.load(landmarks_path)
    else:
        raise ValueError("Unsupported file format for landmarks.")
    
    labels = np.load(labels_path)
    return landmarks, labels

def encode_labels(labels):
    """
    Encodes string labels into integers.
    
    :param labels: List of string labels.
    :return: Tuple (encoded_labels, classes).
    """
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    return encoded_labels, le.classes_

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plots a confusion matrix to visualize model performance.
    
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param classes: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encode_labels(labels):
    """Encode string labels into integers."""
    le = LabelEncoder()
    return le.fit_transform(labels), le.classes_

def build_model(input_shape, num_classes):
    """Build a Feedforward Neural Network (FNN) model."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the model with early stopping and learning rate scheduling."""
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[early_stopping, lr_scheduler])
    return history

def save_model(model, save_path):
    """Save the trained model in HDF5 format."""
    model.save(save_path)

if __name__ == "__main__":
    # Example usage (assuming data is already prepared)
    X_train = np.load('path/to/landmarks_train.npy')
    y_train = np.load('path/to/labels_train.npy')
    X_val = np.load('path/to/landmarks_val.npy')
    y_val = np.load('path/to/labels_val.npy')
    y_train, classes = encode_labels(y_train)
    y_val, _ = encode_labels(y_val)
    input_shape = (X_train.shape[1],)
    num_classes = len(classes)
    model = build_model(input_shape, num_classes)
    train_model(model, X_train, y_train, X_val, y_val)
    save_model(model, 'path/to/save_model.h5')
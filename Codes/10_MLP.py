# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:11:11 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 16:07:50 2025

@author: H.A.R
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import os
import time

# Check for GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Define output directory
output_dir = 'D:/apap1/MLP_Model/'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data = pd.read_csv('D:/apap1/4_Species.csv')
X = data.iloc[:, :-1].values
y = data['Species'].values - 1  # Convert labels to 0-based indexing

# Split data: 90% for cross-validation, 10% for final testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to categorical
num_classes = len(np.unique(y))
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Define MLP model with Leaky ReLU and L2 Regularization
def build_mlp(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.01)),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        layers.Dense(32, kernel_regularizer=regularizers.l2(0.01)),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Early stopping
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 10-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'training_time': []}
conf_matrices = []
histories = []

best_val_acc = 0.0
best_model_path = os.path.join(output_dir, "Best_MLP_Model.h5")

for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"Training Fold {fold+1}/10")
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train_cat[train_index], y_train_cat[val_index]
    
    # Train model
    model = build_mlp(X_train.shape[1], num_classes)
    start_time = time.time()
    history = model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=64, 
                        validation_data=(X_val_fold, y_val_fold), verbose=0, 
                        callbacks=[early_stopping])
    training_time = time.time() - start_time
    histories.append(history.history)
    
    # Save model
    model_path = os.path.join(output_dir, f"MLP_fold{fold+1}.h5")
    model.save(model_path)
    
    # Predictions
    y_pred = model.predict(X_val_fold)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val_fold, axis=1)
    
    # Compute metrics
    acc = accuracy_score(y_val_classes, y_pred_classes)
    metrics['accuracy'].append(acc)
    metrics['precision'].append(precision_score(y_val_classes, y_pred_classes, average='macro'))
    metrics['recall'].append(recall_score(y_val_classes, y_pred_classes, average='macro'))
    metrics['f1'].append(f1_score(y_val_classes, y_pred_classes, average='macro'))
    metrics['mcc'].append(matthews_corrcoef(y_val_classes, y_pred_classes))
    metrics['training_time'].append(training_time)
    
    # Store confusion matrix
    conf_matrix = confusion_matrix(y_val_classes, y_pred_classes)
    conf_matrices.append(conf_matrix)
    np.save(os.path.join(output_dir, f"Confusion_Matrix_fold{fold+1}.npy"), conf_matrix)
    
    # Update best model
    if acc > best_val_acc:
        best_val_acc = acc
        model.save(best_model_path)

# Compute average confusion matrix
avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)
np.save(os.path.join(output_dir, "Confusion_Matrix_Avg.npy"), avg_conf_matrix)

# Compute average training and validation accuracy
max_epochs = max(len(h['accuracy']) for h in histories)
histories = [
    {key: np.pad(h[key], (0, max_epochs - len(h[key])), 'edge') for key in h}
    for h in histories
]
avg_train_acc = np.mean([h['accuracy'] for h in histories], axis=0)
avg_val_acc = np.mean([h['val_accuracy'] for h in histories], axis=0)

np.save(os.path.join(output_dir, "Training_Accuracy.npy"), avg_train_acc)
np.save(os.path.join(output_dir, "Validation_Accuracy.npy"), avg_val_acc)
np.save(os.path.join(output_dir, "Histories.npy"), histories)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(avg_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=1, linecolor='black', square=True, annot_kws={"size": 20})
plt.title('Confusion Matrix - MLP', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.savefig(os.path.join(output_dir, "Confusion_Matrix_Avg.png"))
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(avg_train_acc, label='Training Accuracy', linewidth=2, marker='o', color='blue')
plt.plot(avg_val_acc, label='Validation Accuracy', linewidth=2, marker='s', color='green')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Training and Validation Accuracy - MLP', fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Accuracy_Curve.png"))
plt.show()

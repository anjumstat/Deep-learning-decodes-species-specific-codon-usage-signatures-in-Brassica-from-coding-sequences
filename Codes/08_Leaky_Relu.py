# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 12:06:45 2025

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Leaky ReLU Network - Updated to match DBN/RBFN output format
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
from tensorflow.keras import layers, models, callbacks
import os
import time

# Check for GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Define output directory - same as DBN/RBFN
output_dir = 'E:/Bras/LeakyReLU_updated/'
os.makedirs(output_dir, exist_ok=True)

# Load dataset - same as DBN/RBFN
data = pd.read_csv('E:/Bras/4_Species.csv')
X = data.iloc[:, :-1].values
y = data['Species'].values - 1  # Convert labels to 0-based indexing

# Split data: 90% for cross-validation, 10% for final testing - same as DBN/RBFN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

# Normalize features - same as DBN/RBFN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to categorical - same as DBN/RBFN
num_classes = len(np.unique(y))
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Define Leaky ReLU Model - similar structure to DBN
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(128, input_shape=(input_shape,)),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(0.2),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(0.2),
        layers.Dense(32),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

# Early stopping - same as DBN/RBFN
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 10-Fold Cross-Validation - same structure as DBN/RBFN
kf = KFold(n_splits=10, shuffle=True, random_state=42)
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'training_time': []}
conf_matrices = []
histories = []

best_val_acc = 0.0
best_model_path = os.path.join(output_dir, "Best_LeakyReLU_Model.h5")

for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"Training Fold {fold+1}/10")
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train_cat[train_index], y_train_cat[val_index]
    
    # Train model
    model = build_model(X_train.shape[1], num_classes)
    start_time = time.time()
    history = model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=64, 
                        validation_data=(X_val_fold, y_val_fold), verbose=0, 
                        callbacks=[early_stopping])
    training_time = time.time() - start_time
    histories.append(history.history)
    
    # Save model
    model_path = os.path.join(output_dir, f"LeakyReLU_fold{fold+1}.h5")
    model.save(model_path)
    
    # Predictions
    y_pred = model.predict(X_val_fold)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val_fold, axis=1)
    
    # Compute metrics - same as DBN/RBFN
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

# Compute average metrics after cross-validation - same as DBN/RBFN
avg_metrics = {key: np.mean(value) for key, value in metrics.items()}

# Save metrics as CSV (with average values of the 10 folds) - same as DBN/RBFN
metrics_df = pd.DataFrame(avg_metrics, index=[0])
metrics_df.to_csv(os.path.join(output_dir, "LeakyReLU_Metrics_Avg.csv"), index=False)

# Compute average confusion matrix - same as DBN/RBFN
avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)
np.save(os.path.join(output_dir, "Confusion_Matrix_Avg.npy"), avg_conf_matrix)

# Compute average training and validation accuracy - same as DBN/RBFN
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

# Plot confusion matrix - same style as DBN/RBFN
plt.figure(figsize=(8, 6))
sns.heatmap(avg_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=1, linecolor='black', square=True, annot_kws={"size": 20})
plt.title('Confusion Matrix - Leaky ReLU', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.savefig(os.path.join(output_dir, "Confusion_Matrix_Avg.png"))
plt.show()

# Plot accuracy curves - same style as DBN/RBFN
plt.figure(figsize=(10, 6))
plt.plot(avg_train_acc, label='Training Accuracy', linewidth=2, marker='o', color='blue')
plt.plot(avg_val_acc, label='Validation Accuracy', linewidth=2, marker='s', color='green')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Training and Validation Accuracy - Leaky ReLU', fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Accuracy_Curve.png"))
plt.show()

# =============================================
# TEST DATA EVALUATION WITH CONFUSION MATRIX
# =============================================
# Same structure as DBN/RBFN

# Load the best model
best_model = tf.keras.models.load_model(best_model_path)

# Get predictions on test data
y_test_pred = best_model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Calculate test metrics
test_metrics = {
    'accuracy': accuracy_score(y_test, y_test_pred_classes),
    'precision': precision_score(y_test, y_test_pred_classes, average='macro'),
    'recall': recall_score(y_test, y_test_pred_classes, average='macro'),
    'f1': f1_score(y_test, y_test_pred_classes, average='macro'),
    'mcc': matthews_corrcoef(y_test, y_test_pred_classes)
}

# Generate test confusion matrix
test_conf_matrix = confusion_matrix(y_test, y_test_pred_classes)

# Save test metrics and confusion matrix
pd.DataFrame(test_metrics, index=[0]).to_csv(
    os.path.join(output_dir, "Test_Metrics.csv"), 
    index=False
)
np.save(os.path.join(output_dir, "Test_Confusion_Matrix.npy"), test_conf_matrix)

# Plot enhanced confusion matrix - same style as DBN/RBFN
plt.figure(figsize=(10, 8))
sns.heatmap(test_conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            annot_kws={'size': 16},
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor='grey')

# Customize labels (adjust according to your class names)
class_names = ['Wheat', 'Rice', 'Barley', 'BD']  # Same as DBN/RBFN
plt.xticks(np.arange(len(class_names)) + 0.5, class_names, fontsize=12)
plt.yticks(np.arange(len(class_names)) + 0.5, class_names, fontsize=12, rotation=0)

plt.title('Test Data Confusion Matrix\nLeaky ReLU Network', fontsize=16, pad=20)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.tight_layout()

# Save high-quality figure
plt.savefig(os.path.join(output_dir, "Test_Confusion_Matrix.png"), 
           dpi=300, 
           bbox_inches='tight')
plt.close()

# Display normalized confusion matrix - same as DBN/RBFN
test_conf_matrix_norm = test_conf_matrix.astype('float') / test_conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(test_conf_matrix_norm,
            annot=True,
            fmt='.2f',
            cmap='Purples',
            annot_kws={'size': 12},
            cbar=True,
            square=True)

plt.xticks(np.arange(len(class_names)) + 0.5, class_names, fontsize=12)
plt.yticks(np.arange(len(class_names)) + 0.5, class_names, fontsize=12, rotation=0)
plt.title('Normalized Test Confusion Matrix', fontsize=16, pad=20)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.savefig(os.path.join(output_dir, "Test_Confusion_Matrix_Normalized.png"), 
           dpi=300, 
           bbox_inches='tight')
plt.close()

# Print comprehensive results - same format as DBN/RBFN
print("\n=== Cross-Validation Results ===")
print(f"Average Validation Accuracy: {avg_metrics['accuracy']:.4f}")
print(f"Average Validation Precision: {avg_metrics['precision']:.4f}")
print(f"Average Validation Recall: {avg_metrics['recall']:.4f}")
print(f"Average Validation F1: {avg_metrics['f1']:.4f}")
print(f"Average Validation MCC: {avg_metrics['mcc']:.4f}")

print("\n=== Test Data Results ===")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")
print(f"Test F1-Score: {test_metrics['f1']:.4f}")
print(f"Test MCC: {test_metrics['mcc']:.4f}\n")
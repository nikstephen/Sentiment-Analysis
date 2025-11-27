"""
Comprehensive CNN Model Evaluation
Shows: Confusion Matrix, Precision, Recall, F1 Score, Accuracy, ROC, AUC
"""

import numpy as np
import pandas as pd
import csv
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import matplotlib with backend
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except:
    PLOT_AVAILABLE = False
    print('⚠️  Matplotlib not available')

print("="*80)
print("COMPREHENSIVE CNN MODEL EVALUATION")
print("="*80)

# Load dataset
print("\nLoading dataset...")
file_path = './saved_models/reviews_dataset_1M.txt'

reviews = []
sentiments = []

with open(file_path, 'r', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        reviews.append(row['Review Text'])
        sentiments.append(row['Sentiment'])

# Map sentiments
sentiment_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
y = np.array([sentiment_map[s] for s in sentiments])
X = np.array(reviews)

print(f"✓ Loaded {len(X)} reviews")

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Test samples: {len(X_test)}")

# Load tokenizer
print("\nLoading tokenizer and model...")
with open('./saved_models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Tokenize test data
MAX_LEN = 100
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Load CNN model
model = load_model('./saved_models/cnn_models/cnn_best.keras')
print("✓ Model loaded")

# Make predictions
print("\nMaking predictions...")
y_pred_proba = model.predict(X_test_pad, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate metrics
print("\n" + "="*80)
print("EVALUATION METRICS")
print("="*80)

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Accuracy: {accuracy*100:.2f}%")

# 2. Precision, Recall, F1 Score (per class)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

sentiment_names = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

print("\nPer-Class Metrics:")
print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-" * 55)
for i, name in enumerate(sentiment_names):
    print(f"{name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")

# Weighted averages
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\n{'Weighted Avg':<15} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f}")

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 4. ROC and AUC (one-vs-rest for multiclass)
print("\n" + "="*80)
print("ROC-AUC METRICS (One-vs-Rest)")
print("="*80)

# Binarize labels for ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f"{sentiment_names[i]:<15} AUC: {roc_auc[i]:.4f}")

# Micro-average ROC curve
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print(f"\n{'Micro-average':<15} AUC: {roc_auc['micro']:.4f}")

# Macro-average AUC
roc_auc_macro = roc_auc_score(y_test_bin, y_pred_proba, average='macro')
print(f"{'Macro-average':<15} AUC: {roc_auc_macro:.4f}")

# Classification Report
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_test, y_pred, target_names=sentiment_names))

# Visualizations
if PLOT_AVAILABLE:
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Confusion Matrix Heatmap
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=sentiment_names, yticklabels=sentiment_names)
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax1.set_title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', 
                  fontsize=14, fontweight='bold')
    
    # 2. Confusion Matrix Normalized
    ax2 = plt.subplot(2, 3, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
                xticklabels=sentiment_names, yticklabels=sentiment_names)
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    # 3. Precision, Recall, F1 Score Bar Chart
    ax3 = plt.subplot(2, 3, 3)
    x = np.arange(len(sentiment_names))
    width = 0.25
    
    bars1 = ax3.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax3.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax3.bar(x + width, f1, width, label='F1 Score', color='#e74c3c')
    
    ax3.set_xlabel('Sentiment Class', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Precision, Recall, F1 Score by Class', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sentiment_names)
    ax3.set_ylim([0.95, 1.0])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. ROC Curves
    ax4 = plt.subplot(2, 3, 4)
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    for i, color, name in zip(range(3), colors, sentiment_names):
        ax4.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{name} (AUC = {roc_auc[i]:.4f})')
    
    ax4.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=2,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})')
    
    ax4.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax4.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax4.legend(loc="lower right")
    ax4.grid(alpha=0.3)
    
    # 5. AUC Scores Bar Chart
    ax5 = plt.subplot(2, 3, 5)
    auc_scores = [roc_auc[i] for i in range(3)] + [roc_auc_macro]
    labels = sentiment_names + ['Macro-Avg']
    colors_auc = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    
    bars = ax5.barh(labels, auc_scores, color=colors_auc)
    ax5.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
    ax5.set_title('AUC Scores by Class', fontsize=14, fontweight='bold')
    ax5.set_xlim([0.95, 1.0])
    ax5.grid(axis='x', alpha=0.3)
    
    for i, (bar, score) in enumerate(zip(bars, auc_scores)):
        ax5.text(score + 0.001, i, f'{score:.4f}', 
                va='center', fontweight='bold')
    
    # 6. Overall Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    OVERALL PERFORMANCE SUMMARY
    {'='*40}
    
    Accuracy:              {accuracy*100:.2f}%
    
    Weighted Precision:    {precision_weighted:.4f}
    Weighted Recall:       {recall_weighted:.4f}
    Weighted F1 Score:     {f1_weighted:.4f}
    
    Macro AUC:            {roc_auc_macro:.4f}
    Micro AUC:            {roc_auc['micro']:.4f}
    
    {'='*40}
    Dataset: 45,000 reviews
    Test Set: 9,000 reviews
    Classes: NEGATIVE, NEUTRAL, POSITIVE
    Model: CNN (Convolutional Neural Network)
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.suptitle('CNN Model - Comprehensive Evaluation Metrics', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_file = './saved_models/comprehensive_evaluation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()
else:
    print("\n⚠️  Skipping visualizations (matplotlib not available)")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)

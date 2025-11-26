"""
Train CNN Model Only - With Epochs
High accuracy sentiment analysis for Tagalog/English/Taglish reviews
"""

import numpy as np
import pandas as pd
import csv
import pickle

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import matplotlib after other imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except:
    PLOT_AVAILABLE = False
    print('⚠️  Matplotlib not available - skipping visualizations')

print('='*80)
print('CNN MODEL TRAINING - SENTIMENT ANALYSIS')
print('='*80)
print(f'TensorFlow version: {tf.__version__}\n')

# Load dataset
file_path = './saved_models/reviews_dataset_combined_all.txt'
print(f'Loading dataset: {file_path}')

reviews = []
ratings = []

with open(file_path, 'r', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        reviews.append(row['Review Text'])
        ratings.append(float(row['Rating']))

# Create DataFrame
df = pd.DataFrame({
    'review': reviews,
    'rating': ratings
})

# Map ratings to sentiments
def get_sentiment(rating):
    if rating >= 4:
        return 'POSITIVE'
    elif rating == 3:
        return 'NEUTRAL'
    else:
        return 'NEGATIVE'

df['sentiment'] = df['rating'].apply(get_sentiment)

sentiment_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
df['label'] = df['sentiment'].map(sentiment_map)

print(f'✓ Loaded {len(df)} reviews')
print(f'\nSentiment Distribution:')
print(df['sentiment'].value_counts())

# Split data
X = df['review'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\n✓ Training samples: {len(X_train)}')
print(f'✓ Test samples: {len(X_test)}')

# Tokenization
MAX_WORDS = 10000
MAX_LEN = 100

print(f'\nTokenizing text...')
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

print(f'✓ Vocabulary size: {len(tokenizer.word_index)}')
print(f'✓ Sequences shape: {X_train_pad.shape}')

# Save tokenizer
with open('./saved_models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print('✓ Tokenizer saved\n')

# Build CNN model
print('='*80)
print('BUILDING CNN MODEL')
print('='*80)

cnn_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print('\nCNN Architecture:')
cnn_model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('./saved_models/cnn_models/cnn_best.keras', 
                           save_best_only=True, monitor='val_accuracy', verbose=1)

# Create cnn_models directory
import os
os.makedirs('./saved_models/cnn_models', exist_ok=True)

# Train CNN
print('\n' + '='*80)
print('🚀 TRAINING CNN MODEL')
print('='*80)

history = cnn_model.fit(
    X_train_pad, y_train_cat,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate
print('\n' + '='*80)
print('📊 EVALUATION RESULTS')
print('='*80)

loss, accuracy = cnn_model.evaluate(X_test_pad, y_test_cat, verbose=0)
print(f'\n✓ Test Accuracy: {accuracy*100:.2f}%')
print(f'✓ Test Loss: {loss:.4f}')

# Predictions
y_pred = np.argmax(cnn_model.predict(X_test_pad, verbose=0), axis=1)

print('\n📈 Classification Report:')
print(classification_report(y_test, y_pred, target_names=['NEGATIVE', 'NEUTRAL', 'POSITIVE']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Save model
cnn_model.save('./saved_models/cnn_models/cnn_model.keras')
print('\n✓ CNN model saved to ./saved_models/cnn_models/cnn_model.keras')

# Visualization
if PLOT_AVAILABLE:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training history
    ax1 = axes[0]
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('CNN Training History', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss history
    ax2 = axes[1]
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('CNN Loss History', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Confusion Matrix
    ax3 = axes[2]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
               xticklabels=['NEG', 'NEU', 'POS'],
               yticklabels=['NEG', 'NEU', 'POS'])
    ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax3.set_title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('./saved_models/cnn_training_results.png', dpi=200, bbox_inches='tight')
    print('✓ Visualization saved to ./saved_models/cnn_training_results.png')
    plt.close()
else:
    print('⚠️  Skipping visualization (matplotlib not available)')

# Test samples
print('\n' + '='*80)
print('🧪 TESTING WITH SAMPLE REVIEWS')
print('='*80)

test_samples = [
    "Ang ganda ng product! Highly recommended!",
    "Okay lang naman, pwede na",
    "Pangit! Very disappointing!",
    "Excellent quality and fast delivery!",
    "Average lang, nothing special",
    "Terrible service, waste of money"
]

sentiment_names = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

for i, text in enumerate(test_samples, 1):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = sentiment_names[np.argmax(cnn_model.predict(padded, verbose=0))]
    print(f'{i}. "{text}"')
    print(f'   → Prediction: {pred}\n')

print('='*80)
print('✅ CNN MODEL TRAINING COMPLETE!')
print('='*80)
print(f'\nModel files saved:')
print(f'  • ./saved_models/cnn_models/cnn_model.keras')
print(f'  • ./saved_models/cnn_models/cnn_best.keras')
print(f'  • ./saved_models/tokenizer.pkl')
print(f'  • ./saved_models/cnn_training_results.png')

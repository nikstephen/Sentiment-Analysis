"""
Test Sentiment Analysis - Comprehensive Testing
Tests 30 new samples (10 POSITIVE, 10 NEUTRAL, 10 NEGATIVE)
"""

import numpy as np
import pickle
from tensorflow import keras

print('='*80)
print('COMPREHENSIVE SENTIMENT ANALYSIS TESTING')
print('='*80)

# Load model and tokenizer
print('\nLoading model and tokenizer...')
model = keras.models.load_model('./saved_models/cnn_models/cnn_best.keras')
tokenizer = pickle.load(open('./saved_models/tokenizer.pkl', 'rb'))
print('✓ Model and tokenizer loaded\n')

# Test samples organized by expected sentiment
test_samples = {
    'POSITIVE': [
        "I'm very satisfied with the service. Everything was fast and organized.",
        "Ang ganda ng bagong system! Mas madali nang mag-submit ng requirements.",
        "The teachers were very helpful and responded quickly to my concerns.",
        "Sobrang saya ko sa experience, maayos ang proseso at mababait ang staff.",
        "I love the interface—clean, smooth, and user-friendly.",
        "Napaka-responsive ng support team, agad nilang inayos ang issue ko.",
        "The new update improved the performance a lot. Great job!",
        "Maganda ang feedback mechanism, ramdam ko na pinapakinggan kami.",
        "Everything works perfectly and exceeded my expectations.",
        "Sulit ang effort ng developers, napaka-ganda ng outcome.",
    ],
    'NEUTRAL': [
        "I submitted my request this morning; still waiting for the result.",
        "Ang system minsan gumagana, minsan hindi. Observing pa rin ako.",
        "The update looks the same as before. Nothing major changed.",
        "Nag-log in ako at normal naman ang takbo, wala namang kakaiba.",
        "Some features work fine, others still need improvement.",
        "I received the message, pero hindi ko pa nagagamit ang bagong module.",
        "It's an okay experience—not bad, not great.",
        "The page loads, pero medyo matagal minsan.",
        "Hindi ako sure kung may difference after the patch, trying pa rin.",
        "The design is plain but understandable.",
    ],
    'NEGATIVE': [
        "The system keeps crashing and it's very frustrating.",
        "Sobrang bagal ng proseso, nakaka-inis gamitin.",
        "The staff didn't respond and I felt ignored the whole time.",
        "Ang daming bugs, hindi ko magawa yung kailangan ko.",
        "The interface is confusing and poorly designed.",
        "Hindi stable ang connection, lagi akong na-di-disconnect.",
        "The update made everything worse—mas maraming error ngayon.",
        "Sobrang disappointing, hindi ito naka-meet sa expectations ko.",
        "Puro loading lang kahit mabilis naman internet ko.",
        "The system completely failed during my submission. Sayang oras ko.",
    ]
}

sentiment_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
MAX_LEN = 100

# Test each category
correct_predictions = 0
total_predictions = 0

for expected_sentiment, reviews in test_samples.items():
    print('='*80)
    print(f'TESTING {expected_sentiment} SAMPLES (Expected: {expected_sentiment})')
    print('='*80)
    
    correct_in_category = 0
    
    for i, text in enumerate(reviews, 1):
        # Preprocess
        seq = tokenizer.texts_to_sequences([text])
        padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        
        # Predict
        pred_proba = model.predict(padded, verbose=0)[0]
        pred_class = np.argmax(pred_proba)
        predicted_sentiment = sentiment_classes[pred_class]
        confidence = pred_proba[pred_class] * 100
        
        # Check if correct
        is_correct = (predicted_sentiment == expected_sentiment)
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        
        if is_correct:
            correct_in_category += 1
            correct_predictions += 1
        
        total_predictions += 1
        
        # Display result
        print(f'\n{i}. "{text}"')
        print(f'   Expected: {expected_sentiment} | Predicted: {predicted_sentiment} ({confidence:.2f}%) {status}')
        print(f'   Probabilities: NEG={pred_proba[0]*100:.2f}% | NEU={pred_proba[1]*100:.2f}% | POS={pred_proba[2]*100:.2f}%')
    
    accuracy_in_category = (correct_in_category / len(reviews)) * 100
    print(f'\n✓ {expected_sentiment} Accuracy: {correct_in_category}/{len(reviews)} ({accuracy_in_category:.1f}%)')

# Overall Summary
print('\n' + '='*80)
print('OVERALL SUMMARY')
print('='*80)
print(f'\nTotal samples tested: {total_predictions}')
print(f'Correct predictions: {correct_predictions}')
print(f'Wrong predictions: {total_predictions - correct_predictions}')
print(f'\n🎯 Overall Accuracy: {(correct_predictions/total_predictions)*100:.2f}%')

if correct_predictions == total_predictions:
    print('\n🎉 PERFECT SCORE! All predictions are correct!')
elif (correct_predictions/total_predictions) >= 0.9:
    print('\n⭐ EXCELLENT! Very high accuracy!')
elif (correct_predictions/total_predictions) >= 0.8:
    print('\n👍 GOOD! Strong performance!')
else:
    print('\n⚠️  Needs improvement. Some predictions were incorrect.')

print('\n' + '='*80)
print('✅ TESTING COMPLETE')
print('='*80)

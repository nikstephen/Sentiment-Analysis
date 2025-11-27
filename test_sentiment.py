"""
Test Sentiment Analysis - Batch Testing
Tests multiple reviews and shows predictions with confidence scores
"""

import numpy as np
import pickle
from tensorflow import keras

print('='*80)
print('SENTIMENT ANALYSIS TESTING')
print('='*80)

# Load model and tokenizer
print('\nLoading model and tokenizer...')
model = keras.models.load_model('./saved_models/cnn_models/cnn_best.keras')
tokenizer = pickle.load(open('./saved_models/tokenizer.pkl', 'rb'))
print('✓ Model and tokenizer loaded\n')

# Test samples
test_reviews = [
    "Minsan hindi fair ang grading, dapat mas transparent.",
    "I feel comfortable in this school because everyone is friendly.",
    "Maganda yung pagtuturo pero minsan ang bilis ng lessons, hindi nakakasabay yung iba.",
    "The school facilities are okay, but the internet is always slow",
    "Sobrang bait ng mga teachers namin, lalo na sa pagtulong kapag nahihirapan kami",
    "I think the school needs more extracurricular activities for students.",
    "Minsan crowded sa hallway, sana ayusin ang traffic flow ng students.",
    "The lessons are clear and organized. I appreciate the effort of the instructors.",
    "Sana mas strict sa cleanliness lalo na sa comfort rooms.",
    "The school events are fun, pero sana mas madalas.",
    "Nakaka-stress minsan kasi sabay-sabay ang requirements.",
    "Our teacher explains well and gives many examples that help us understand.",
    "Mas okay sana kung may mas maraming electric fans sa classroom.",
    "The school environment is peaceful and good for learning.",
    "Sana huwag masyadong maraming quizzes sa isang araw.",
    "I like how the teachers motivate us to do better.",
    "Medyo luma na yung mga computers sa lab, sana ma-upgrade soon.",
    "I hate you",
    "Minsan hindi organized ang announcements kaya nakakalito.",
    "The library is very helpful, complete yung resources.",
    "Sana may mas maraming chairs sa cafeteria lalo na tuwing lunch time.",
    "Our adviser is very supportive and approachable.",
    "Madalas mainit sa room, sana ayusin ang ventilation.",
    "I enjoy the activities because they are interactive and fun.",
    "Hindi ko gusto yung schedule kasi masyadong late yung uwi namin.",
    "Sana may mas maraming school clubs para sa iba't ibang interests.",
    "Minsan hindi fair ang grading, dapat mas transparent.",
    "I feel comfortable in this school because everyone is friendly.",
    "Sana bawasan ang written works, masyado nang madami.",
    "Sana mas maaga magbigay ng announcement para makapag-prepare.",
    "Sana may free printing service for students.",
    "Medyo strict ang ibang teachers, pero understandable naman.",
    "Sometimes the lessons feel rushed.",
    "Thankful ako kasi marami akong natutunan this quarter.",
    "Sana mas i-improve ang campus security.",
    "Minsan walang aircon kaya nahihirapan mag-focus.",
    "Medyo mabagal ang process sa office, sana mas mapa-bilis."
]

sentiment_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
MAX_LEN = 100

# Test each review
print('='*80)
print('TESTING RESULTS')
print('='*80)

# Count sentiments
sentiment_counts = {'NEGATIVE': 0, 'NEUTRAL': 0, 'POSITIVE': 0}

for i, text in enumerate(test_reviews, 1):
    # Preprocess
    seq = tokenizer.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    
    # Predict
    pred_proba = model.predict(padded, verbose=0)[0]
    pred_class = np.argmax(pred_proba)
    sentiment = sentiment_classes[pred_class]
    confidence = pred_proba[pred_class] * 100
    
    # Count
    sentiment_counts[sentiment] += 1
    
    # Display result
    print(f'\n{i}. "{text}"')
    print(f'   Sentiment: {sentiment} ({confidence:.2f}%)')
    print(f'   Probabilities: NEG={pred_proba[0]*100:.2f}% | NEU={pred_proba[1]*100:.2f}% | POS={pred_proba[2]*100:.2f}%')

# Summary
print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f'\nTotal reviews tested: {len(test_reviews)}')
print(f'\nSentiment Distribution:')
print(f'  POSITIVE:  {sentiment_counts["POSITIVE"]:2d} ({sentiment_counts["POSITIVE"]/len(test_reviews)*100:.1f}%)')
print(f'  NEUTRAL:   {sentiment_counts["NEUTRAL"]:2d} ({sentiment_counts["NEUTRAL"]/len(test_reviews)*100:.1f}%)')
print(f'  NEGATIVE:  {sentiment_counts["NEGATIVE"]:2d} ({sentiment_counts["NEGATIVE"]/len(test_reviews)*100:.1f}%)')

print('\n' + '='*80)
print('✅ TESTING COMPLETE')
print('='*80)

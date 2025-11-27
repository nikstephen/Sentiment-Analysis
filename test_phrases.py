import tensorflow as tf
import pickle
import numpy as np

# Load model and tokenizer
model = tf.keras.models.load_model('./saved_models/cnn_models/cnn_best.keras')
tokenizer = pickle.load(open('./saved_models/tokenizer.pkl', 'rb'))

# Test phrases
phrases = [
    'ang gara naman ng pamamalakad sa school na ito ayoko ng ganito!!',
    'dami kong nakitang mali sa payment ayusin nyo ito lalo yung transaction ng payment.',
    'its amazing to see how this system works but there are minor defects'
]

# Tokenize and pad
sequences = tokenizer.texts_to_sequences(phrases)
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100, padding='post')

# Predict
predictions = model.predict(padded, verbose=0)
sentiments = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

print('\n' + '='*80)
print('SENTIMENT ANALYSIS - TEST PHRASES')
print('='*80 + '\n')

for i, phrase in enumerate(phrases):
    pred_idx = np.argmax(predictions[i])
    print(f'{i+1}. "{phrase}"')
    print(f'   Predicted: {sentiments[pred_idx]} ({predictions[i][pred_idx]*100:.2f}%)')
    print(f'   Probabilities: NEG={predictions[i][0]*100:.2f}% | NEU={predictions[i][1]*100:.2f}% | POS={predictions[i][2]*100:.2f}%')
    print()

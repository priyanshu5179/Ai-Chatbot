import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import pickle
import random
from nltk_utils import tokenize, stem, bag_of_words

# Load intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Prepare training data
all_words = []
tags = []
xy = []

# Loop through each sentence in intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = tokenize(pattern)
        all_words.extend(w)
        # Add to xy pair
        xy.append((w, intent['tag']))
        # Add to tags list
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# Stem and lower each word, remove duplicates
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"{len(xy)} patterns")
print(f"{len(tags)} tags: {tags}")
print(f"{len(all_words)} unique stemmed words: {all_words}")

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    # Y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(f"Input size: {input_size}, Output size: {output_size}")

# Build Neural Network
model = Sequential()
model.add(Dense(hidden_size, input_shape=(input_size,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_size, activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
print("Training...")
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)
print("Training complete!")

# Save model and data
model.save('chatbot_model.h5')
print("Model saved to chatbot_model.h5")

# Save words and classes for later use
data = {
    "words": all_words,
    "tags": tags
}
pickle.dump(data, open("chatbot_data.pkl", "wb"))
print("Data saved to chatbot_data.pkl")
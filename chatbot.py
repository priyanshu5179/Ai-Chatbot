import json
import numpy as np
import pickle
from keras.models import load_model
from nltk_utils import bag_of_words, tokenize

class ChatBot:
    def __init__(self):
        # Load model and data
        self.model = load_model('chatbot_model.h5')
        data = pickle.load(open('chatbot_data.pkl', 'rb'))
        self.words = data['words']
        self.tags = data['tags']
        
        # Load intents for responses
        with open('intents.json', 'r') as f:
            self.intents = json.load(f)
    
    def predict_class(self, sentence):
        # Return list of intents with probability
        bow = bag_of_words(tokenize(sentence), self.words)
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        
        # Filter predictions with threshold
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        # Sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        for r in results:
            return_list.append({
                'intent': self.tags[r[0]],
                'probability': str(r[1])
            })
        return return_list
    
    def get_response(self, intents_list):
        if not intents_list:
            # No prediction above threshold
            tag = 'unknown'
        else:
            tag = intents_list[0]['intent']
        
        # Find response for predicted tag
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = np.random.choice(i['responses'])
                break
        return result, tag
    
    def chat(self, message):
        ints = self.predict_class(message)
        res, tag = self.get_response(ints)
        return res, tag

# Test in terminal
if __name__ == "__main__":
    print("Loading chatbot...")
    bot = ChatBot()
    print("Chatbot ready! Type 'quit' to exit.")
    
    while True:
        message = input("You: ")
        if message.lower() == 'quit':
            break
        
        response, tag = bot.chat(message)
        print(f"Bot ({tag}): {response}")
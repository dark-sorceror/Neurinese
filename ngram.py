from preprocess import translate
from collections import defaultdict, Counter

class ngram:
    def __init__(self):
        self.model = defaultdict(Counter)
        
    def fit(self, text: str):
        words = text.split()
        
        for i in range(len(words) - 2):
            w1 = words[i]
            w2 = words[i + 1]
            w3 = words[i + 2]
            
            self.model[(w1, w2)][w3] += 1
        
        
    def predict(self, w1: str, w2: str):
        return self.model[(w1, w2)].most_common(1)[0][0]

sentences = """
The weather is very good
The weather is very good today
"""

ngram = ngram()
ngram.fit(sentences)

input_sentence = "天气非常"

translated = translate(input_sentence, True).split()

pred = ngram.predict(translated[-2], translated[-1])

print(pred)
print(translate(pred, False)[0]) # Take first character out of phrase for now
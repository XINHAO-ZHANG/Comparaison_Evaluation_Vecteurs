# from collections import Counter
# import re

# # Assume `corpus` is a list of documents. For example:
# # corpus = ['text of document one', 'text of document two', 'text of document three']

# corpus_path="./Comparaison_Evaluation_Vecteurs/corpus/pride_and_prejudice.txt"
# # Join all documents into one large string for easier processing
# #corpus_string = ' '.join(corpus)
# with open(corpus_path, 'r') as f:
#     corpus_string = f.read().replace('\n', '')

# # Tokenize the string into words. This will split words and remove punctuation.source 
# # You might need to adapt the regex to your specific needs.
# words = re.findall(r'\b\w+\b', corpus_string.lower())

# # Count the frequency of each word in the corpus
# word_counts = Counter(words)

# # Get the 100 most common words
# most_common_words = word_counts.most_common(100)

# # Print the 50 most common words with their frequencies
# for word, freq in most_common_words:
#     print(f"{word}: {freq}")

# import nltk
# from nltk.corpus import stopwords
# from collections import Counter
# import re

from collections import Counter
# NLTK models
import nltk
from collections import Counter
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

# Assume `corpus` is a list of documents. For example:
# corpus = ['text of document one', 'text of document two', 'text of document three']

corpus_path = "./Comparaison_Evaluation_Vecteurs/corpus/pride_and_prejudice.txt"
stopwords = nltk.corpus.stopwords.words('english')

# Lire le corpus
with open(corpus_path, 'r') as f:
    corpus_string = f.read().replace('\n', ' ')

# Tokenize le string par phrases
sentences = nltk.sent_tokenize(corpus_string)

# Tokenize each sentence into words and tag part of speech
words = []
for sentence in sentences:
    words.extend(nltk.pos_tag(nltk.word_tokenize(sentence)))

# Filter out stopwords and non-alphabetic words, and get only nouns (NN) and adjectives (JJ)
filtered_words = [word for word, pos in words if (pos in ['NN', 'JJ']) and (word.isalpha() and word not in stopwords)]

# Count the frequency of each noun and adjective in the corpus
word_counts = Counter(filtered_words)

# Get the 25 most common nouns and adjectives
most_common_words = word_counts.most_common(29)

# Print the 25 most common nouns and adjectives with their frequencies
for word, freq in most_common_words:
    print(f"{word}: {freq}")


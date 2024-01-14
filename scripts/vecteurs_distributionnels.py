import csv
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from collections import Counter
from scipy.sparse import csr_matrix

# Téléchargez les ressources nécessaires de nltk
nltk.download('punkt')
nltk.download('stopwords')

############################## prétraitement du texte ###############################
def preprocess_text(file_path):
    # Lire le fichier texte et le convertir en minuscules
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    # Tokenize et supprimez les stop-words et les mots non alphabétiques (telles que les ponctuations)
    tokens = word_tokenize(text)
    tokens_nettoyé = [token for token in tokens if token.isalpha() and token not in stopwords.words('english')]
    return tokens_nettoyé

############################## Matrice de co-occurrence #############################
def co_occurrence_matrix(tokens, vocab, window_size=5):
    vocab_index = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)
    # Utilisez une matrice creuse pour économiser de la mémoire
    co_occurrence = csr_matrix((vocab_size, vocab_size), dtype=np.float32)
    for i, token in enumerate(tokens):
        token_index = vocab_index.get(token, -1)
        if token_index != -1:
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(tokens))
            for j in range(start, end):
                if i != j:
                    context_word = tokens[j]
                    context_index = vocab_index.get(context_word, -1)
                    if context_index != -1:
                        co_occurrence[token_index, context_index] += 1.0
    return co_occurrence


############################### Matrice PPMI ########################################
from scipy.sparse import coo_matrix

def ppmi_matrix(co_occurrence, epsilon=1e-8):
    # Convertissez la matrice de co-occurrence en une matrice dense pour le calcul de PPMI
    # Nous ajoutons epsilon pour éviter la division par zéro dans le cas où expected est nul
    co_occurrence_dense = co_occurrence.toarray() + epsilon
    total = co_occurrence_dense.sum()
    sum_over_rows = co_occurrence_dense.sum(axis=1)
    sum_over_cols = co_occurrence_dense.sum(axis=0)
    expected = np.outer(sum_over_rows, sum_over_cols) / total
    
    # Utilisez la formule PPMI, mais opérez sur les matrices denses
    ppmi_dense = np.maximum(np.log2(co_occurrence_dense / expected), 0)
    
    # Convertissez la matrice PPMI dense en une matrice creuse pour économiser de la mémoire
    ppmi = coo_matrix(ppmi_dense)
    return ppmi


# Chemin vers le fichier corpus
file_path = './corpus/pride_and_prejudice.txt'

# Prétraitement du texte
tokens = preprocess_text(file_path)

# Créez un vocabulaire basé sur les mots les plus fréquents
vocab_size = 10000 
vocab = [word for word, count in Counter(tokens).most_common(vocab_size)]

# Construction de la matrice de co-occurrence
co_occurrence = co_occurrence_matrix(tokens, vocab)

# Calcul de la matrice PPMI
ppmi = ppmi_matrix(co_occurrence)


############################## comparaison qualitative ##############################

# Normalisez les vecteurs PPMI
ppmi_normalized = normalize(ppmi, norm='l2', axis=1)

# Mots-cibles pour la comparaison
target_words = ['sister', 'time', 'much', 'little', 'good', 'nothing', 'family', 'man', 'dear', 'great', 'mother', 'father', 'day', 'young', 'last', 'letter', 'room', 'friend', 'first', 'way', 'house', 'sure', 'manner', 'pleasure', 'aunt']
k = 10  

# Path where the CSV file will be saved
csv_file_path = './résultats/vecteurs_distributionnels.csv'
# Open the CSV file in write mode
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header
    csv_writer.writerow(['Target Word', 'Neighbors'])

    # Write neighbors for each target word
    for word in target_words:
        if word in vocab:
            word_index = vocab.index(word)
            word_vector = ppmi_normalized[word_index].reshape(1, -1)
            cos_similarity = cosine_similarity(word_vector, ppmi_normalized)
            neighbors_indices = cos_similarity.argsort()[0][-k-1:-1][::-1]
            neighbors = [vocab[idx] for idx in neighbors_indices]
            csv_writer.writerow([word, ', '.join(neighbors)])


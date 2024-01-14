import csv
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from collections import Counter
from scipy.sparse import csr_matrix

# Téléchargez les ressources nécessaires de nltk
nltk.download('punkt')
nltk.download('stopwords')

############################## prétraitement du texte ##############################
def preprocess_text(file_path):
    # Lire le fichier texte et le convertir en minuscules
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    # Tokenize et supprimez les stop-words et les mots non alphabétiques (telles que les ponctuations)
    tokens = word_tokenize(text)
    tokens_nettoyé = [token for token in tokens if token.isalpha() and token not in stopwords.words('english')]
    return tokens_nettoyé

############################## Matrice de co-occurrence ##############################
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


############################## Matrice PPMI ##############################
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

# chemin du fichier texte
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

##########################################################################################
# B: Réduction de dimension  PCA

# Définir le nombre de dimensions souhaitées après la réduction
n_components = 2  # Par exemple, réduire à 100 dimensions

# Initialiser l'objet PCA
pca = PCA(n_components=n_components)

# Adapter le PCA à la matrice PPMI normalisée et la transformer
ppmi_reduced_pca = pca.fit_transform(ppmi.toarray())

####### t-SNE ####################
# Définir le nombre de dimensions souhaitées après la réduction
n_components_tsne = 2  

# Initialiser l'objet t-SNE
tsne = TSNE(n_components=n_components_tsne, perplexity=30, learning_rate=200, n_iter=1000, verbose=1, random_state=42)

# Adapter le t-SNE à la matrice PPMI normalisée et la transformer
ppmi_reduced_tsne = tsne.fit_transform(ppmi.toarray())


############################## comparaison qualitative #####################################
# Normalisez les vecteurs PPMI
ppmi_normalized_pca = normalize(ppmi_reduced_pca, norm='l2', axis=1)
ppmi_normalized_tsne = normalize(ppmi_reduced_tsne, norm='l2', axis=1)

# Mots-cibles pour la comparaison
target_words = ['sister', 'time', 'much', 'little', 'good', 'nothing', 'family', 'man', 'dear', 'great', 'mother', 'father', 'day', 'young', 'last', 'letter', 'room', 'friend', 'first', 'way', 'house', 'sure', 'manner', 'pleasure', 'aunt']
k = 10  

with open('./résultats/méthodeB_réduction_dimensionalité.csv', 'w', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    
    csv_writer.writerow(['PCA', 'word voisins'])  # En-tête du CSV pour PCA
    
    for word in target_words:
        word_index = vocab.index(word)
        word_vector_pca = ppmi_normalized_pca[word_index].reshape(1, -1)
        cos_similarity_pca = cosine_similarity(word_vector_pca, ppmi_normalized_pca)
        neighbors_indices_pca = cos_similarity_pca.argsort()[0][-k-1:-1][::-1]
        neighbors_pca = [vocab[idx] for idx in neighbors_indices_pca]
        csv_writer.writerow([word, neighbors_pca])  # Écrire les données dans le CSV
    
    csv_writer.writerow(['t-SNE', 'word voisins'])  # En-tête du CSV pour t-SNE
    
    for word in target_words:
        word_index = vocab.index(word)
        word_vector_tsne = ppmi_normalized_tsne[word_index].reshape(1, -1)
        cos_similarity_tsne = cosine_similarity(word_vector_tsne, ppmi_normalized_tsne)
        neighbors_indices_tsne = cos_similarity_tsne.argsort()[0][-k-1:-1][::-1]
        neighbors_tsne = [vocab[idx] for idx in neighbors_indices_tsne]
        csv_writer.writerow([word, neighbors_tsne])  # Écrire les données dans le CSV
    
    print(f"{word}: {neighbors_pca}")
    print(f"{word}: {neighbors_tsne}")

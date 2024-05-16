import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

text = "Verilen metin örneği. Başka bir cümle örneği de var. Geldim ve döndüm. Baktım yoktun. Geldim eve sana. Yardım et. Aç kapıyı. Okula gittim. Sensin o. Yalvarma bana. Nereliyim."
sentences = sent_tokenize(text)

print(sentences)

from py2neo import Graph, Node

# Neo4j veritabanına bağlanma
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Her cümle için bir node oluşturma
for sentence in sentences:
    node = Node("Sentence", text=sentence)
    graph.create(node)

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string

# Stop-word listesini yükleme
stop_words = set(stopwords.words('english'))

# Tokenization, Stemming, Stop-word Elimination ve Punctuation işlemlerini uygulama
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

processed_sentences = []
for sentence in sentences:
    # Cümleyi küçük harflere dönüştürme
    sentence = sentence.lower()
    # Punctuation işlemi
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(sentence)
    # Stop-word Elimination
    tokens = [token for token in tokens if token not in stop_words]
    # English Stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]

    processed_sentence = ' '.join(lemmatized_tokens)
    processed_sentences.append(processed_sentence)

print(processed_sentences)

from gensim.models import Word2Vec

# Tokenlara ayrılmış cümleleri kullanarak Word2Vec modelini eğitme
model = Word2Vec([sentence.split() for sentence in processed_sentences], min_count=1)


import numpy as np

# Cümlelerin vektörlerini oluşturma
sentence_vectors = [np.mean([model.wv[token] for token in sentence.split()], axis=0) for sentence in processed_sentences]

# Kosinüs benzerliği hesaplama
similarity_matrix = np.zeros((len(processed_sentences), len(processed_sentences)))
for i in range(len(processed_sentences)):
    for j in range(len(processed_sentences)):
        similarity_matrix[i][j] = np.dot(sentence_vectors[i], sentence_vectors[j]) / (np.linalg.norm(sentence_vectors[i]) * np.linalg.norm(sentence_vectors[j]))


import networkx as nx
import matplotlib.pyplot as plt

# Grafı oluşturma
G = nx.Graph()

# Her cümle için bir düğüm (node) ekleme
for i in range(len(sentences)):
    G.add_node(i, text=sentences[i])

# Benzerlik matrisindeki değerlere dayanarak bağlantıları ekleme
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        similarity_score = round(similarity_matrix[i][j], 2) # Benzerlik skorunu 2 haneyle sınırla
        print(similarity_score)
        G.add_edge(i, j, weight=similarity_score)

# Grafı çizme
pos = nx.circular_layout(G) # Dairesel düzen
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.show()
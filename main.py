import docx2txt
import nltk
from masaustu import dokuman_yukle
import dokuman_modulu


from neo4j import GraphDatabase
from nltk.tokenize import sent_tokenize

uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(username, password))



my_text = dokuman_modulu.dokuman_icerik
#print(my_text)

metin = dokuman_modulu.dokuman_icerik

# Belirli bir satırdan başlayarak metni al
baslangic_satiri = 1  # 0'dan başlamak için 0 olarak ayarlayabilirsiniz
metin = metin.split('\n')[baslangic_satiri:]

my_string = ' '.join(metin)
my_string=my_string.replace("\t", "")

sentences = sent_tokenize(my_string)


baslik = docx2txt.process("Örnek Doküman.docx")

ilk_satir = baslik.split('\n')[0]

ilk_baslik = sent_tokenize(ilk_satir)

print(ilk_baslik)








print("İlk Satır:", ilk_satir)


#print(sentences)
#print(len(sentences))

from py2neo import Graph, Node, Relationship
graph = Graph(uri, auth=(username, password))

for sentence in sentences:
    node = Node("Sentence", text=sentence)
    graph.merge(node, "Sentence", "text")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        node_a = Node("Sentence", text=sentences[i])
        node_b = Node("Sentence", text=sentences[j])
        graph.merge(node_a, "Sentence", "text")
        graph.merge(node_b, "Sentence", "text")
        relation = Relationship(node_a, "c", node_b)
        graph.merge(relation)  # İlişkiyi merge ile oluşturuyoruz

# Dolandırarak bağlantıları gösterme
result = graph.run("""
MATCH (a:Sentence)-[:SIMILAR_TO]->(b:Sentence)
RETURN a.text, b.text
""")



from nltk.tokenize import word_tokenize

tokenized_sentences = []
for sentence in sentences:
    tokens = word_tokenize(sentence)
    tokenized_sentences.append(tokens)

baslik_tokenized_sentences = []
for ilk_sentence in ilk_baslik:
    baslik_tokens = word_tokenize(ilk_sentence)
    baslik_tokenized_sentences.append(baslik_tokens)

#print(baslik_tokenized_sentences)



#print(tokenized_sentences)
from nltk.stem import PorterStemmer

baslik_stemmed_sentences = []

stemmed_sentences = []


from nltk.stem import SnowballStemmer
import string
stemmer = SnowballStemmer('english')

punctuations = set(string.punctuation)

for sentence_tokens in tokenized_sentences:
    stemmed_tokens = [stemmer.stem(token) for token in sentence_tokens if token not in punctuations]
    stemmed_sentences.append(stemmed_tokens)

for baslik_sentence_tokens in baslik_tokenized_sentences:
    baslik_stemmed_tokens = [stemmer.stem(token) for token in baslik_sentence_tokens if token not in punctuations]
    baslik_stemmed_sentences.append(baslik_stemmed_tokens)

#print(baslik_stemmed_sentences)

import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
filtered_sentences = []
for sentence_tokens in stemmed_sentences:
    filtered_tokens = [token for token in sentence_tokens if token.lower() not in stop_words]
    filtered_sentences.append(filtered_tokens)

baslik_filtered_sentences = []
for baslik_sentence_tokens in baslik_stemmed_sentences:
    baslik_filtered_tokens = [token for token in baslik_sentence_tokens if token.lower() not in stop_words]
    baslik_filtered_sentences.append(baslik_filtered_tokens)

#print(baslik_filtered_sentences)

import string

punctuations = set(string.punctuation)
final_sentences = []

for sentence_tokens in filtered_sentences:
    final_tokens = [token for token in sentence_tokens if token not in punctuations]
    final_sentences.append(final_tokens)

baslik_final_sentences = []

for baslik_sentence_tokens in baslik_filtered_sentences:
    baslik_final_tokens = [token for token in baslik_sentence_tokens if token not in punctuations]
    baslik_final_sentences.append(baslik_final_tokens)

#print(baslik_final_sentences)



from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity



# Word2Vec modelini eğit
def train_word2vec(sentences):
    model = Word2Vec(sentences, min_count=1, epochs=epochs)  # epochs parametresini ekledik
    return model


# Cümleler arasındaki benzerliği hesapla
def calculate_similarity(model, sentence1, sentence2):
    vector1 = sum([model.wv.get_vector(word) for word in sentence1]) / len(sentence1)
    vector2 = sum([model.wv.get_vector(word) for word in sentence2]) / len(sentence2)
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity




epochs = 50  # Eğitim tekrar sayısını artırabilirsiniz
model = train_word2vec(final_sentences)




for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sentence1 = final_sentences[i]
        sentence2 = final_sentences[j]
        similarity = calculate_similarity(model, sentence1, sentence2)
        print("Benzerlik:", similarity, "Cümle 1:", final_sentences[i], "Cümle 2:", final_sentences[j])





def ozel_isim_kontrolu(cumle):
    # Cümledeki özel isimleri tanımlamak için bir Named Entity Recognizer (NER) kullanıyoruz.
    # nltk.download('punkt')  # Eğer daha önce indirmediyseniz, bir kez çalıştırmanız gerekebilir.
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')
    tokens = nltk.word_tokenize(cumle)
    tagged = nltk.pos_tag(tokens)
    named_entities = nltk.ne_chunk(tagged, binary=True)

    # Cümledeki özel isim sayısını hesapla
    ozel_isim_sayisi = sum(1 for chunk in named_entities if hasattr(chunk, 'label') and chunk.label() == 'NE')

    # Özel isim kontrolünü hesapla
    ozel_isim_kontrolu = ozel_isim_sayisi / len(tokens)

    return ozel_isim_kontrolu

def cumleleri_ayir(metin):
    cumleler = nltk.sent_tokenize(metin)
    ozel_isim=[]
    for cumle in cumleler:
        ozel_isim_kontrol_sonucu = ozel_isim_kontrolu(cumle)
        ozel_isim.append(ozel_isim_kontrol_sonucu)
    return ozel_isim


ozel_isim_skor = cumleleri_ayir(my_text)


import re

def numerik_veri_kontrolu(cumle):
    # Cümledeki numerik veri sayısını hesapla
    numerik_veri_sayisi = len(re.findall(r'\d+', cumle))

    # Numerik veri kontrolünü hesapla
    numerik_veri_kontrolu = numerik_veri_sayisi / len(cumle.split())

    return numerik_veri_kontrolu

def cumleleri_numerik_ayir(metin):
    cumleler = nltk.sent_tokenize(metin)
    numerik_veri=[]
    for cumle in cumleler:
        numerik_veri_kontrol_sonucu = numerik_veri_kontrolu(cumle)
        numerik_veri.append(numerik_veri_kontrol_sonucu)
    return numerik_veri

numerik_veri_skor = cumleleri_numerik_ayir(my_string)



stringbaslik = " ".join([str(eleman) if isinstance(eleman, str) else " ".join(eleman) for eleman in baslik_final_sentences])
print(stringbaslik)

stringcumleler = " ".join([str(eleman) if isinstance(eleman, str) else " ".join(eleman) for eleman in final_sentences])

tokenized_baslik = word_tokenize(stringbaslik)
tokenized_baslik = [kelime.lower() for kelime in tokenized_baslik]

tokenized_cumleler = [[kelime.lower() for kelime in cumle] for cumle in final_sentences]
print(tokenized_cumleler)

def baslik_kontrol(baslik, cumle):
    baslik_kelimeleri = set(baslik)
    cumle_kelimeleri = set(cumle)
    if len(cumle_kelimeleri) == 0:
        return 0.0  # Sıfıra bölme hatasını önlemek için sıfır döndür
    kesisim = baslik_kelimeleri.intersection(cumle_kelimeleri)
    return len(kesisim) / len(cumle_kelimeleri)

# Test
baslik_veri=[]
for cumle in tokenized_cumleler:
    kontrol = baslik_kontrol(tokenized_baslik, cumle)
    baslik_veri.append(kontrol)

   # print("Cümle:", " ".join(cumle))
    #print("Kontrol Sonucu:", kontrol)
    #print("----------")
print(baslik_veri)

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [' '.join(sentence) for sentence in final_sentences]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

theme_words = set()
for i in range(tfidf_matrix.shape[0]):
    sentence = final_sentences[i]
    scores = {feature_names[col]: tfidf_matrix[i, col] for col in tfidf_matrix[i, :].nonzero()[1]}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    threshold = len(sentence) // 10
    for word, score in sorted_scores:
        if word in sentence and word not in theme_words:
            theme_words.add(word)
            if len(theme_words) >= threshold:
                break

tema_veri=[]

# Cümlenin içinde geçen tema kelime sayısının cümlenin uzunluğuna bölünmesi
for i in range(len(sentences)):
    sentence = final_sentences[i]
    theme_word_count = sum(1 for word in sentence if word in theme_words)
    theme_word_ratio = theme_word_count / len(sentence)
    tema_veri.append(theme_word_ratio)
    #print(f"Cümle {i+1}: {sentences[i]}")
    #print(f"Tema Kelime Oranı: {theme_word_ratio}\n")
print(tema_veri)
#cumleleri_numerik_ayir(my_string)

#cumleleri_ayir(my_string)



liste1 = ozel_isim_skor
liste2 = numerik_veri_skor
liste3 = baslik_veri
liste4 = tema_veri

toplam_listesi = []
for i in range(len(liste1)):
    toplam = liste1[i] + liste2[i] + liste3[i] + liste4[i]
    toplam_listesi.append(toplam)
print(toplam_listesi)

def metin_ozetleme(metin, cümle_skorları, özet_cümle_sayısı):
    cümle_listesi = metin.split(". ")
    cümle_sayısı = min(len(cümle_listesi), len(cümle_skorları))

    cümle_sıralaması = sorted(range(cümle_sayısı), key=cümle_skorları.__getitem__, reverse=True)

    özet_metin = []
    for cümle_sırası in cümle_sıralaması[:özet_cümle_sayısı]:
        özet_metin.append(cümle_listesi[cümle_sırası])

    return '. '.join(özet_metin)



import math

cümle_skorları = toplam_listesi
özet_cümle_sayısı = len(sentences)/2
ozet_tamsayi = math.floor(özet_cümle_sayısı)
net_cumle= " ".join(my_string.split())
özet_metin = metin_ozetleme(net_cumle, cümle_skorları, ozet_tamsayi)
print(özet_metin)


def calculate_rouge_1_score(reference, summary):
    reference_words = reference.split()
    summary_words = summary.split()

    intersection = len(set(reference_words) & set(summary_words))
    reference_length = len(reference_words)

    rouge_1_score = intersection / reference_length

    return rouge_1_score

reference_text = docx2txt.process("özet.docx")
summary_text = özet_metin
print(reference_text)

rouge_1_score = calculate_rouge_1_score(reference_text, summary_text)
print("ROUGE-1 skoru:", rouge_1_score)
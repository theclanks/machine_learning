
# coding: utf-8

# In[372]:

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#import seaborn

#%matplotlib inline

from utils import tokenizer
from wordcloud import WordCloud
import nltk
from nltk import FreqDist

from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import tokenize    
from math import log
from os.path import basename, splitext
from os import listdir
import json


# In[32]:

def plot_word_cloud(words, filename=None):
    wordcloud = WordCloud(
        width=800,
        height=600,
        max_words=500,
        scale=3,
    )

    wordcloud.generate_from_frequencies(dict(words))
    plt.figure(figsize=(15,20))
    plt.imshow(wordcloud)
    plt.axis("off")
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    
    plt.show()


# In[262]:

def carregar_arquivo(arquivo):
    base = basename(arquivo)
    genero = splitext(base)[0]
    print(genero)
    genero_df = pd.read_csv(arquivo)
    return (genero_df, genero)


# In[559]:

def carregar_arquivo_json(arquivo):
    base = basename(arquivo)
    genero = splitext(base)[0]
    print(genero)
    with open(arquivo) as f:
        genero_df = json.load(f)
    return (genero_df, genero)


# In[862]:

def gerar_stopwords():
    STOP_WORDS = set(nltk.corpus.stopwords.words('portuguese'))
    
    for w in nltk.corpus.stopwords.words('portuguese'):
        STOP_WORDS.add(w.capitalize())
    
    STOP_WORDS.add("pra")
    STOP_WORDS.add("t")
    STOP_WORDS.add("E")
    STOP_WORDS.add(u"é")
    STOP_WORDS.add(u"É")
    STOP_WORDS.add(u"...")
    STOP_WORDS.add(u"''")
    STOP_WORDS.add(u"``")
    STOP_WORDS.add(u"2x")
    STOP_WORDS.add(u"A")
    STOP_WORDS.add(u"O")
    return STOP_WORDS


# In[327]:

def remove_stopwords(freq_dist):
    
    # remove punctuation and stopwords
    for stopword in STOP_WORDS:
        if stopword in freq_dist:
            del freq_dist[stopword]

    for punctuation in tokenizer.CHARACTERS_TO_SPLIT:
        if punctuation in freq_dist:
            del freq_dist[punctuation]
    return freq_dist


# In[334]:

def distribuicao_frequencia(arquivo):
    genero_df, genero = carregar_arquivo(arquivo)

    genero_all = " ".join(genero_df.lyric.values).decode('utf8')
    palavras_tokenize = tokenize.word_tokenize(genero_all, language='portuguese')
    
    genero_freq_dist = FreqDist(palavras_tokenize)
    genero_freq_dist = remove_stopwords(genero_freq_dist)
    return genero_freq_dist, genero
    
    


# In[336]:

def word_cloud(genero_freq_dist, genero):
    plot_word_cloud(genero_freq_dist.most_common(200), "resources/full-" + genero + "-wordcloud.png")


# In[333]:

def distribuicao_palavras(genero_freq_dist, genero):
    genero_word_frequencies = [x[1] for x in genero_freq_dist.most_common(200000)]
    plt.plot(genero_word_frequencies)

    plt.yscale("log")
    plt.xscale("log")

    plt.title("Word Frequencies", fontsize=16)
    plt.ylabel("Word Count", fontsize=14)
    plt.xlabel("Number of Words", fontsize=14)

    plt.savefig("resources/"+ genero + "-word-distribution.png")
    


# In[346]:

def words_corpus_ranking(genero_freq_dist, portugues_freq_dist):
    
    #Remover palavras raras
    portugues_freq_dist = {k:v for k,v in portugues_freq_dist.items() if v > 5}
    n_portugues = sum(portugues_freq_dist.values())
    
    genero_freq_dist = {k:v for k,v in genero_freq_dist.items() if v > 5}
    n_genero = sum(genero_freq_dist.values())
    
    # combinar
    genero_rank = {}

    for w in genero_freq_dist.keys():
        if w in portugues_freq_dist.keys():
            if len(w) > 2:
                genero_rank[w] = log( (float(genero_freq_dist[w]) / float(n_genero)) / (float(portugues_freq_dist[w]) / float(n_genero)))
    return genero_rank


# In[248]:

def list_to_markdown_table(lst, headings):
    assert(len(headings) == len(lst[0]))
    results = ""
    results += "|".join(headings) + "\n"
    results += "|".join(["---"] * len(headings)) + "\n"
    #print lst
    for elem in lst:
        results += "|"
        results += str(elem[0])
        results += "|"
        results += elem[1].encode('utf8')
        results += "|"
        results += str(elem[2])
        results += "\n"
        
    return results


# In[863]:

STOP_WORDS = gerar_stopwords()


# In[344]:

portugues = nltk.corpus.mac_morpho.words() + nltk.corpus.floresta.words()


# In[345]:

portugues_freq_dist = FreqDist([w.lower() for w in portugues
                              if w not in STOP_WORDS])


# In[ ]:




# ### Carregando letras

# In[ ]:




# In[373]:

for f in listdir("data/raw"):
    arquivo = "data/raw/"+f
    genero_freq_dist, genero = distribuicao_frequencia(arquivo)
    word_cloud(genero_freq_dist, genero)
    distribuicao_palavras(genero_freq_dist, genero)
    genero_rank = words_corpus_ranking(genero_freq_dist, portugues_freq_dist)
    
    sorted_ranked_genero = [(x[0], x[1][0], "{:.2f}".format(x[1][1])) for x in
                       enumerate(sorted(genero_rank.items(), key=lambda x: -x[1]))]
    
    headings = ["Rank", "Word", "Score"]
    
    top_10 = list_to_markdown_table(sorted_ranked_genero[:10:1], headings)
    botton_10 = list_to_markdown_table(sorted_ranked_genero[:-10:-1], headings)
    print(top_10)
    
    with open("data/" + genero + "_rank.json", "w") as f:
        json.dump(genero_rank, f)


# In[ ]:




# In[347]:

## Sem machine Learning


# In[374]:

def score_genero(s):
    words = tokenizer.tokenize_strip_non_words(s)
    
    if len(words) == 0:
        return 0
    
    score = 0
    for word in words:
        score += metalness.get(word,0)
        
    return score / len(words)


# In[ ]:




# In[653]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.utils import Bunch
from sklearn.naive_bayes import MultinomialNB


# In[699]:

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[597]:

# Com Stemming caiu um pouco
#nltk.download('rslp')


# In[ ]:




# In[ ]:

# Criar um dataset


# In[792]:

dataset = None
data = []
target = []


# In[793]:

def pre_process(lyric):
    palavras_tokenize = tokenize.word_tokenize(lyric.decode('utf8'), language='portuguese')
    genero_freq_dist = FreqDist(palavras_tokenize)
    genero_freq_dist = remove_stopwords(genero_freq_dist)
    #genero_freq_dist = map(stemm.stem, genero_freq_dist.keys())
    return ' '.join(genero_freq_dist.keys())
    


# In[794]:

# Diretamente das musicas
for f in listdir("data/raw"):
    arquivo = "data/raw/"+f
    
    genero_df, genero = carregar_arquivo(arquivo)
    #stemm = nltk.stem.RSLPStemmer()
    
    for l in genero_df.values:
        normalized = pre_process(l[0])
        data.append(normalized)
        target.append(genero)

dataset = Bunch(data=data, target=target)

    


# In[733]:

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.30, random_state=42)


# In[ ]:




# In[734]:

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)


# In[735]:

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[ ]:




# In[736]:

teste = ["Olha que coisa mais linda, mais cheia de graça, moça do corpo dourado"] # Bossa


# In[737]:

teste = ['Borboletinha, está na cozinha'] # Bossa + -


# In[738]:

teste = ["Tudo que tu fazes, fazes muito bem, a cada toque teu,a cada amanhecer"] # Gospel


# In[ ]:




# In[740]:

text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()),])


# In[741]:

text_clf = text_clf.fit(X_train, y_train)


# In[742]:

scores = cross_val_score(text_clf, X_test, y_test, cv=5, scoring='f1_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
preds = text_clf.predict(X_test)
print(metrics.classification_report(y_test, preds))


# In[744]:

# Benchmark
text_clf.predict(teste) # Gospel


# In[ ]:




# In[598]:

## Ajustes


# In[745]:

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
}


# In[746]:

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(dataset.data, dataset.target)


# In[747]:

scores = cross_val_score(gs_clf, X_test, y_test, cv=5, scoring='f1_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
preds = gs_clf.predict(X_test)
print(metrics.classification_report(y_test, preds))


# In[750]:

# Benchmark
gs_clf.predict(teste) # Gospel


# In[ ]:




# In[ ]:




# In[807]:

text_clf_mlp = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-mlp', MLPClassifier()),
                ])

text_clf_mlp.fit(X_train, y_train)


# In[752]:

scores = cross_val_score(text_clf_mlp, X_test, y_test, cv=5, scoring='f1_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
preds = text_clf_mlp.predict(X_test)
print(metrics.classification_report(y_test, preds))


# In[756]:

# Benchmark
text_clf_mlp.predict(teste) # Gospel


# In[ ]:




# In[ ]:




# In[757]:

text_clf_rf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-rf', RandomForestClassifier()),
                ])

text_clf_rf.fit(X_train, y_train)


# In[758]:

scores = cross_val_score(text_clf_rf, X_test, y_test, cv=5, scoring='f1_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
preds = text_clf_rf.predict(X_test)
print(metrics.classification_report(y_test, preds))


# In[759]:

# Benchmark
text_clf_rf.predict(teste) # Gospel


# In[ ]:




# In[770]:

get_ipython().magic(u'pinfo SGDClassifier')


# In[771]:

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='modified_huber', penalty='l2',
                                            alpha=1e-3, max_iter=5,random_state=42)),
])

text_clf_svm.fit(X_train, y_train)


# In[772]:

scores = cross_val_score(text_clf_svm, X_test, y_test, cv=5, scoring='f1_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
preds = text_clf_svm.predict(X_test)
print(metrics.classification_report(y_test, preds))


# In[786]:

# Benchmark
text_clf_svm.predict(teste) # Gospel


# In[ ]:




# In[774]:

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf-svm__alpha': (1e-2, 1e-3),
}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X_train, y_train)
print(gs_clf_svm.best_score_)


# In[775]:

scores = cross_val_score(gs_clf_svm, X_test, y_test, cv=5, scoring='f1_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
preds = gs_clf_svm.predict(X_test)
print(metrics.classification_report(y_test, preds))


# In[776]:

# Benchmark
gs_clf_svm.predict(teste) # Gospel


# In[ ]:




# In[787]:

print("Logloss")
print("MLP: %s" % metrics.log_loss(y_test, text_clf_mlp.predict_proba(X_test)))
print("SVM: %s" % metrics.log_loss(y_test, text_clf_svm.predict_proba(X_test)))
print("SVM+: %s" % metrics.log_loss(y_test, gs_clf_svm.predict_proba(X_test)))
print("NB: %s" % metrics.log_loss(y_test, text_clf.predict_proba(X_test)))
print("RF: %s" % metrics.log_loss(y_test, text_clf_rf.predict_proba(X_test)))


# In[860]:

# Salvar modelo
# A técnica de NB foi a que menos "decorou" o conjunto de dados, ao testar com outras músicas 
# extraidas do site letras.com.br acertou a grande maioria.


# In[828]:

import joblib


# In[859]:

joblib.dump(text_clf, "md.pkl", compress=True)


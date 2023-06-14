from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

from nltk.stem import *
from nltk import word_tokenize
import pymorphy2


def read_data_xlsx(file):
    """Принимает на вход имя Excel файла и возвращает DataFrame.
    file - Имя файла
    """
    df = pd.read_excel(file)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    return df

def stemmer(texts):
    """Функция принимающая список текстов и возвращающая его после стемминга
    texts - исходный текст
    stem_texts - список лемматизированных текстов
    """
    porter_stemmer = PorterStemmer()
    stem_texts = []
    for text in texts:
        text = text.lower()
        nltk_tokens = word_tokenize(text)
        line = ''
        for word in nltk_tokens:
            line += ' ' + porter_stemmer.stem(word)
        stem_texts.append(line)
    return stem_texts


#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#_morph = pymorphy2.MorphAnalyzer()
_lemmatizer = WordNetLemmatizer()    # для английского текста
def lemmatize(texts):
    """Функция лемматизации для списка текстов
    text - исходный текст
    res - список лемматизированных текстов
    """
    res = list()
    for text in texts:
        text = text.lower()
        nltk_tokens = word_tokenize(text) # разбиваем текст на слова
        line = ''
        for word in nltk_tokens:
            #parse = _morph.parse(word)[0]  # Это было для русских слов
            # дальше обрабатываем все части речи
            lemm_word = _lemmatizer.lemmatize(word, pos='n')
            lemm_word = _lemmatizer.lemmatize(lemm_word, pos='v')
            lemm_word = _lemmatizer.lemmatize(lemm_word, pos='r')
            lemm_word = _lemmatizer.lemmatize(lemm_word, pos='a')
            lemm_word = _lemmatizer.lemmatize(lemm_word, pos='s')
            line += ' ' + lemm_word
        res.append(line) # lemmatize для английских слов
    return res


_morph = pymorphy2.MorphAnalyzer()
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
_lemmatizer = WordNetLemmatizer()    # для английского текста
def lemmatize_rus(texts):
    """ Функция лемматизации. Возвращает список лемм слов из исходного текста, только для русского текста
    texts - список текстов 
    """
    res = list()
    for text in texts:
        text = text.lower()
        nltk_tokens = word_tokenize(text) # разбиваем текст на слова
        line = ''
        for word in nltk_tokens:
            parse = _morph.parse(word)[0]
            lemm_word = lemmatize([parse.normal_form])[0] # lemmatize для английских слов
            line += ' ' + lemm_word
        res.append(line) # lemmatize для английских слов
    return res


def multiclassifier(X_train, X_test, Y_train, Y_test, clf, parameters, max_features=None, stop_words='english'):
    """Классификация в зависимости от поданной на вход модели
    X_train - обучающая выборка
    X_test - тестовая выборка
    Y_train - метки обучающей выборки
    Y_test - метки тестовой выборки
    clf - медель классификатора
    parameters - параметры для настройки GridSearchCV
    Возвращает список принадлежности к классам, отчет классификации и лучшие параметры
    return test_pred, , classification_report gs_clf.best_params_
    """
    # Подгонка модели логистической регрессии
    # clf = LogisticRegression(random_state=0)
    text_clf = Pipeline([('vect', CountVectorizer(max_features=max_features, stop_words=stop_words)),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', clf,)])
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=2, cv=3)

    gs_clf.fit(X_train, Y_train)
    prediction = gs_clf.predict(X_test)

    classificationReport = classification_report(Y_test, prediction)
    
    return prediction, classificationReport, gs_clf.best_params_

def classifier(X_train, X_test, Y_train, Y_test, clf, parameters):
    """Классификация в зависимости от поданной на вход модели
    X_train - обучающая выборка
    X_test - тестовая выборка
    Y_train - метки обучающей выборки
    Y_test - метки тестовой выборки
    clf - медель классификатора
    parameters - параметры для настройки GridSearchCV
    Возвращает список принадлежности к классам, отчет классификации и лучшие параметры
    return test_pred, , classification_report gs_clf.best_params_
    """
    # Подгонка модели логистической регрессии
    # clf = LogisticRegression(random_state=0)
    gs_clf = GridSearchCV(clf, parameters, n_jobs=2, cv=3)
    gs_clf.fit(X_train, Y_train)
    prediction = gs_clf.predict(X_test)

    classificationReport = classification_report(Y_test, prediction)
    
    return prediction, classificationReport, gs_clf.best_params_


def pca(data, target, title=''):
    plt.figure()
    data_r_2 = PCA(n_components=2, random_state=0)
    data_reduced_2 = data_r_2.fit_transform(data)
    plt.scatter(data_reduced_2[:, 0], data_reduced_2[:, 1], c=target,
                # cmap=mcolors.ListedColormap(["red", "gray"]),
                edgecolor="k",
                s=40)
    plt.title(title) 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    data_r_3 = PCA(n_components=3, random_state=0)
    data_reduced_3 = data_r_3.fit_transform(data)
    ax.scatter(data_reduced_3[:, 0], data_reduced_3[:, 1], data_reduced_3[:, 2], c=target,
                # cmap=mcolors.ListedColormap(["red", "gray"]),
                edgecolor="k",
                s=40)
    plt.title(title) 

    return data_reduced_2, data_reduced_3



def tsne(data, target, n_components=2, perplexity=30, learning_rate='auto', n_iter=1000, metric='euclidean', init='random', early_exaggeration=12.0):

    # Применение t-SNE для визуализации данных в двух измерениях
    tsne_ = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration,
                 learning_rate=learning_rate, n_iter=n_iter, metric=metric, init=init
                 )
    X_tsne = tsne_.fit_transform(data)

    # Визуализация результата
    if n_components == 2:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target,
                    # cmap=mcolors.ListedColormap(["red", "gray"]),
                    edgecolor="k",
                    s=40)
    if n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=target,
                    # cmap=mcolors.ListedColormap(["red", "gray"]),
                    edgecolor="k",
                    s=40)
    plt.show()
# -*- coding: utf-8 -*-

# import
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



def read_data_xlsx(file):
    """Принимает на вход имя Excel файла и возвращает DataFrame.
    file - Имя файла
    """
    df = pd.read_excel(file)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    return df


_morph = pymorphy2.MorphAnalyzer()
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
_lemmatizer = WordNetLemmatizer()    # для английского текста
def lemmatize(text):
    """Функция лемматизации. Возвращает список лемм слов из исходного текста, только для русского текста
    text - исходный текст
    """
    words = text.split() # разбиваем текст на слова
    res = list()
    for word in words:
        parse = _morph.parse(word)[0]
        res.append(_lemmatizer.lemmatize(parse.normal_form)) # lemmatize для английских слов
    return res

# Создание переменных со стоп словами
#nltk.download('stopwords')     # загружаются единожды C:\Users\Igorexy\AppData\Roaming\nltk_data
_stopwords_rus = stopwords.words('russian')
_stopwords_en = stopwords.words('english')

def del_stop_words(data):
    """Функция удаляет стоп-слова из списка текстов
    data - list[] текстовых данных
    """
    data_output = []
    for i in range(len(data)):
        text = str(data[i]).lower()
        # удаление стоп слов
        for stopw in _stopwords_rus + _stopwords_en:
            if text.find(stopw + ' ') == 0 or text.find(stopw + ', ') == 0:
                text = text.replace(stopw + ' ', "", 1)
                text = text.replace(stopw + ', ', "", 1)
            text = text.replace(' ' + stopw + ' ', " ")
            text = text.replace(' ' + stopw + ',', " ")
            text = text.replace(' ' + stopw + '.', "")
            text = text.replace(' ' + stopw + '!', "")
            text = text.replace(' ' + stopw + '?', "")
        # лемматизация текста (приведение слова к лемме)
        lemm_list = lemmatize(text)   # Для русского и английского текста
        lemm_text = ' '.join(lemm_list)
        data_output.append(lemm_text)
    return data_output

def document_term_matrix(texts, tf_idf=True):
    """Функция создает документ-термин матрицу
    texts - список текстов
    tf_idf=True - Указывает на то, использовать меру TF-IDF или нет
    Выходные параметры: матрица документ-термин, термины
    """
    if tf_idf:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    else:
        vectorizer = CountVectorizer(ngram_range=(1, 2))
    doc_term = vectorizer.fit_transform(texts)
    return doc_term.toarray(), vectorizer.get_feature_names_out()

def choce_from_data(data, tf_idf=True):
    """Возвращает из DataFrame data имена, аннотации в виде листа [tf-idf матрица, термины] и лейблы.
    Не является универсальной функцией, подходит под конкретную выборку"""
    name = del_stop_words(data['name'])
    annotation = del_stop_words(data['annotation'])
    name_doc_term = document_term_matrix(texts=name, tf_idf=tf_idf)
    annotation_doc_term = document_term_matrix(texts=annotation, tf_idf=tf_idf)
    # преобразовываем из строки в лист и берем нужный элемент(лейбл) в цикле по всей выборке
    labels = [eval(data['classification_labels'][i])[0]['flag'] for i in range(len(data['classification_labels']))]
    return name_doc_term, annotation_doc_term, labels

def reshape_testdata(test, terms_train, terms_test):
    """Переформирование тестовой выборки под обучающую.
    Функция ищет совпадающие термины в тестовой выборки и переформировывает под обучающую
    test - тестовая матрица документ-термин
    terms_train - термины обучающей выборки
    terms_test - термины тестовой выборки
    """
    output_test = np.zeros((test.shape[0], len(terms_train)))
    for i in range(len(terms_train)):
        if terms_train[i] in terms_test:
            index = list(terms_test).index(terms_train[i])
            for j in range(output_test.shape[0]):
                output_test[j][i] = test[j][index]  # j-ая строка, i-й индекс
    return output_test



func_list__ = [name for (name , obj) in vars().items()
                 if hasattr(obj, "__class__") and obj.__class__.__name__ == "function"]



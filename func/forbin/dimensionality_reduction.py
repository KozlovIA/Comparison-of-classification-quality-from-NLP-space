import pandas as pd

def dimensionality_reduction_func(data_tf_idf_term):
    """Функция для удаления малозначимых терминов из матрицы tf-idf.
    Переменная limit отвечает за количество встреченных одинаковых терминов, если их меньше limit, то мы их удаляем.
    data_tf_idf_term - список, где нулевой элемент - матрица tf-idf и первый элемент - список терминов
    Возвращает DataFrame с tf-idf матрицей и терминами в заголовках"""
    # Подсчет количества терминов в документах
    term_frequency_name = [[], []]
    for j in range(len(data_tf_idf_term[1])):
        term_frequency_name[0].append(data_tf_idf_term[1][j])
        term_frequency_name[1].append(0)
        for i in range(len(data_tf_idf_term[0])):
            if data_tf_idf_term[0][i][j] != 0:
                term_frequency_name[1][j] += 1


    # Подсчет в каких тестах встречались термины
    above = 0; below = 0
    limit = 3
    term_del_name = []   # Список терминов для удаления
    for i in range(len(term_frequency_name[0])):
        if term_frequency_name[1][i] >= limit:
            above += 1
        else:
            below += 1
            term_del_name.append(term_frequency_name[0][i])

    # print("Названия:")
    # print("Терминов больше порогового значения: " + str(above) + "\nТерминов меньше порогового значения: " + str(below) + "\nПороговое значение: " + str(limit))

    # Удаление малозначимых терминов
    df_tf_idf_short = pd.DataFrame(data_tf_idf_term[0], columns=list(data_tf_idf_term[1]))
    df_tf_idf_short.drop(columns=term_del_name, axis=1, inplace=True)

    return df_tf_idf_short


def df_to_list(df_tf_idf_short):
    """Получение терминов и tf-idf отдельно на основе функции dimensionality_reduction_func"""
    # Получение матрицы tf-idf с сокращенной размерностью
    tf_idf_short = []

    for i in range(len(df_tf_idf_short)):
        tf_idf_short.append(list(df_tf_idf_short.iloc[i]))

    terms = list(df_tf_idf_short.keys())

    return tf_idf_short, terms


def read_dimData_fromFiles(tf_idf_file, terms_file):
    """Считывание tf-idf матриц и терминов из текстовых файлов"""
    # Считывание tf_idf_short из файла
    tf_idf_short = []
    file = open(tf_idf_file, 'r', encoding='utf-8')
    for line in file:
        tf_idf_short.append(eval(line))
    file.close()
    
    # Считывание terms_by_name_short из файла
    file = open(terms_file, 'r', encoding='utf-8')
    terms_short = eval(file.read())
    file.close()

    return tf_idf_short, terms_short

"""# Запись annotation_tf_idf_short в файл, для более быстрого получения переменной
file = open("annotation_tf_idf_short.txt", 'w', encoding='utf-8')
for i in range(len(annotation_tf_idf_short)):
    file.write(str(annotation_tf_idf_short[i]))
    if i != len(annotation_tf_idf_short)-1:
        file.write('\n')
file.close()"""

"""# Считывание annotation_tf_idf_short из файла
annotation_tf_idf_short_FromFile = []
file = open("annotation_tf_idf_short.txt", 'r', encoding='utf-8')
for line in file:
    annotation_tf_idf_short_FromFile.append(eval(line))
file.close()"""


"""# Считывание terms_by_annatation_short из файла
file = open("terms_by_annatation_short.txt", 'r', encoding='utf-8')
terms_by_annatation_short = eval(file.read())
file.close()"""
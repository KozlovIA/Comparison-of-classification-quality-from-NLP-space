from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import time


def pca(data, target, title=''):
    plt.figure()
    data_r_2 = PCA(n_components=2, random_state=0)
    data_reduced_2 = data_r_2.fit_transform(data)
    plt.scatter(data_reduced_2[:, 0], data_reduced_2[:, 1], c=target,
                cmap=mcolors.ListedColormap(["red", "gray"]),
                edgecolor="k",
                s=40)
    plt.title(title) 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    data_r_3 = PCA(n_components=3, random_state=0)
    data_reduced_3 = data_r_3.fit_transform(data)
    ax.scatter(data_reduced_3[:, 0], data_reduced_3[:, 1], data_reduced_3[:, 2], c=target,
                cmap=mcolors.ListedColormap(["red", "gray"]),
                edgecolor="k",
                s=40)
    plt.title(title) 
    # print("Выделенные компоненты", data_r_2.components_, "Главные компоненты(2)", data_r_2.explained_variance_ratio_, sep='\n')
    # print("Главные компоненты(3)", data_r_3.explained_variance_ratio_, sep='\n')



def logistic_regression(X_train, X_test, Y_train, Y_test, label=""):
    """Алгоритм логистической регрессии с ROC кривой и визуализацией предсказанных данных
    X_train - обучающая выборка
    X_test - тестовая выборка
    Y_train - метки обучающей выборки
    Y_test - метки тестовой выборки
    Возвращает список принадлежности к классам, точность, матрицу ошибок, отчет классификации
    return test_pred, test_score, confusion_matrix, classification_report
    """
    # Подгонка модели логистической регрессии
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, Y_train)

    # Прогнозирование результатов тестового набора и вычисление точности.
    test_pred = clf.predict(X_test)

    # Точность классификации на тестовом наборе (Accuracy of logistic regression classifier on test set)
    test_score = clf.score(X_test, Y_test)
    
    confusionMatrix = confusion_matrix(list(Y_test), test_pred)
    
    classificationReport = classification_report(Y_test, test_pred)

    logit_roc_auc = roc_auc_score(Y_test, clf.predict(X_test))
    fpr, tpr, thresholds = roc_curve(Y_test, clf.predict_proba(X_test)[:,1])
    ROC_curve = plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + label)
    plt.legend(loc="lower right")

    pca(X_test, test_pred, title='Logistic Regression ' + label)


    return test_pred, test_score, confusionMatrix, classificationReport





def k_nearest_neighbors(X_train, X_test, Y_train, Y_test, label):  # k ближайших соседей
    """
    Алгоритм k-ближайших соседей
    X_train - обучающая выборка
    X_test - тестовая выборка
    Y_train - метки обучающей выборки
    Y_test - метки тестовой выборки
    Возвращает classification_report, confusion_matrix, accuraccy
    """
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, Y_train)
    Y = neigh.predict(X_test)

    logit_roc_auc = roc_auc_score(Y_test, neigh.predict(X_test))
    fpr, tpr, thresholds = roc_curve(Y_test, neigh.predict_proba(X_test)[:,1])
    ROC_curve = plt.figure()
    plt.plot(fpr, tpr, label='KNeighbors (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + label)
    plt.legend(loc="lower right")

    pca(X_test, Y, title='KNeighbors PCA' + label)

    return Y, classification_report(Y_test, Y), confusion_matrix(Y_test, Y), accuracy_score(Y_test, Y)

def random_tree(X_train, X_test, y_train, y_test, label='', param={'criterion': ["gini", "entropy"],
                          'max_depth': list(range(100,201,10)),
                          'max_features': list(range(0,101,10))}):
    """Классификация с помощью дерева решений"""

    start_time = time.time()

    model = GridSearchCV(DecisionTreeClassifier(),
                         param,
                         n_jobs=3, cv=5)
    model.fit(X_train, y_train)

    y_proba = model.predict(X_test)

    pca(X_test, y_proba, title='Decision Tree Classifier ' + label)

    print('Обучение модели')
    print('Best_params random_tree ' + label)
    print(model.best_params_)

    print("Decision Tree for " + label, (time.time() - start_time)//60, "min", (time.time() - start_time)%60, "sec")
    
    return y_proba, classification_report(y_test, y_proba), confusion_matrix(y_test, y_proba), accuracy_score(y_test, y_proba)


def random_forest(X_train, X_test, y_train, y_test, label='', param={'criterion': ["gini", "entropy"],
                          'n_estimators':list(range(10,101,10)),
                          'max_depth': list(range(10,151,10)),
                          'max_features': list(range(10,151,10))}):
    """Классификация с помощью случайного леса"""
    model = GridSearchCV(RandomForestClassifier(),
                         param,
                         n_jobs=3, cv=5)

    start_time = time.time()

    model.fit(X_train, y_train)
    y_proba = model.predict(X_test)
    
    pca(X_test, y_proba, title='Random Forest Classifier ' + label)

    print('Обучение модели')
    print('Best_params Random Forest Classifier ' + label)
    print(model.best_params_)
    print('Report')
    print(classification_report(y_test, model.predict(X_test)))

    print("Random Forest for " + label, (time.time() - start_time)//60, "min", (time.time() - start_time)%60, "sec")

    return y_proba, classification_report(y_test, y_proba), confusion_matrix(y_test, y_proba), accuracy_score(y_test, y_proba)



# Автоматическое сохранение результатов классификации в таблицу
import docx

def autoTable(classificationReport__, path=''):
    """Автосохранение classificationReport в таблицу"""
    classificationReport__ = classificationReport__.split()
    doc = docx.Document()

    table = doc.add_table(rows=6, cols=5)
    table.style = 'Table Grid'

    k=0
    for i in range(0, 6):
        j=0
        while j < 5:
            if i == 0 and j == 0:
                j+=1
                continue
            if classificationReport__[k-1] == "accuracy":
                j = j + 2
            if classificationReport__[k] == "macro" or classificationReport__[k] == "weighted":
                k+=1
                classificationReport__[k] = classificationReport__[k-1] + " " + classificationReport__[k]
            cell = table.cell(i, j)
            cell.text = classificationReport__[k]
            k+=1; j+=1
    doc.save(path + "classificationReport.docx") 
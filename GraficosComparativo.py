import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from time import time
from scipy.sparse import csr_matrix

# Carregar o novo arquivo CSV
file_path = "dados.csv"
df = pd.read_csv(file_path)

# Mostrar os cinco primeiros registros
print('\nMostrando os 5 primeiros registros:')
print(df.head(5))

# Mostrar as informações do DataFrame
print('\nMostrando as informações do DataFrame:')
df.info()

# Mostrar Sentimentos
print('\nMostrando Sentimentos:')
print(df['SENTIMENTO'].value_counts())

# Dividir os dados em atributos de entrada (x_data) e rótulo de classe (y_data)
x_data = df['PALAVRA_CHAVE']
y_data = df['SENTIMENTO']

# Vetorização de texto usando TF-IDF
# Isso converte o texto em uma representação numérica para uso em algoritmos de aprendizado de máquina.
vectorizer = TfidfVectorizer()
x_data = vectorizer.fit_transform(x_data)

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)


# Função para mostrar desempenho com verificação de dados densos ou esparsos
def mostrar_desempenho(x_train, y_train, x_test, y_test, model, name):
    # Verifica se os dados são esparsos e converter para matriz densa, se necessário
    if isinstance(x_train, csr_matrix):
        x_train = x_train.toarray()
        x_test = x_test.toarray()

    # Treinando o modelo
    inicio = time()
    model.fit(x_train, y_train)
    fim = time()
    tempo_treinamento = (fim - inicio) * 1000

    # Prevendo os dados
    inicio = time()
    y_predicted = model.predict(x_test)
    fim = time()
    tempo_previsao = (fim - inicio) * 1000

    print('Relatório Utilizando Algoritmo', name)
    print('\nMostrando Matriz de Confusão:')
    print(confusion_matrix(y_test, y_predicted))
    print('\nMostrando Relatório de Classificação:')
    print(classification_report(y_test, y_predicted))
    accuracy = accuracy_score(y_test, y_predicted)
    print('Accuracy:', accuracy)
    print('Tempo de treinamento (ms):', tempo_treinamento)
    print('Tempo de previsão (ms):', tempo_previsao)
    return accuracy, tempo_treinamento, tempo_previsao


# Inicializando e avaliando modelos
model_mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10000)
acc_mlp, tt_mlp, tp_mlp = mostrar_desempenho(x_train, y_train, x_test, y_test, model_mlp, 'MLP')

model_arvore = tree.DecisionTreeClassifier()
acc_dt, tt_dt, tp_dt = mostrar_desempenho(x_train, y_train, x_test, y_test, model_arvore, 'DecisionTree')

model_rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
acc_rf, tt_rf, tp_rf = mostrar_desempenho(x_train, y_train, x_test, y_test, model_rf, 'RandomForest')

acc_ada, tt_ada, tp_ada = mostrar_desempenho(x_train, y_train, x_test, y_test, AdaBoostClassifier(), 'AdaBoost')

model_knn = KNeighborsClassifier(n_neighbors=17, p=12, metric='cosine')
acc_knn, tt_knn, tp_knn = mostrar_desempenho(x_train, y_train, x_test, y_test, model_knn, 'KNN')

acc_lr, tt_lr, tp_lr = mostrar_desempenho(x_train, y_train, x_test, y_test, LogisticRegression(), 'LR')

acc_svm, tt_svm, tp_svm = mostrar_desempenho(x_train, y_train, x_test, y_test, SVC(), 'SVM')

acc_gnb, tt_gnb, tp_gnb = mostrar_desempenho(x_train, y_train, x_test, y_test, GaussianNB(), 'GaussianNB')

acc_lda, tt_lda, tp_lda = mostrar_desempenho(x_train, y_train, x_test, y_test, LinearDiscriminantAnalysis(), 'LDA')

model_qda = QuadraticDiscriminantAnalysis()
acc_qda, tt_qda, tp_qda = mostrar_desempenho(x_train, y_train, x_test, y_test, model_qda, 'QDA')

# Comparação de Desempenho em Accuracy
fig = plt.figure(figsize=(15, 10))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
algoritmos = ['GaussianNB', 'MLP', 'DecisionTree', 'KNN', 'Regressão Linear', 'LDA', 'SVM', 'RandomForest', 'AdaBoost',
              'QDA']
accs = [acc_gnb, acc_mlp, acc_dt, acc_knn, acc_lr, acc_lda, acc_svm, acc_rf, acc_ada, acc_qda]
ax.bar(algoritmos, accs)
ax.set_title('Comparação de Desempenho em Accuracy')
plt.show()

# Comparação de Desempenho em Tempo de Treinamento
fig = plt.figure(figsize=(15, 10))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
tts = [tt_gnb, tt_mlp, tt_dt, tt_knn, tt_lr, tt_lda, tt_svm, tt_rf, tt_ada, tt_qda]
ax.bar(algoritmos, tts)
ax.set_title('Comparação de Desempenho em tempo de Treinamento (aprendizado)')
plt.show()

# Comparação de Desempenho em Tempo de Previsão
fig = plt.figure(figsize=(15, 10))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
tps = [tp_gnb, tp_mlp, tp_dt, tp_knn, tp_lr, tp_lda, tp_svm, tp_rf, tp_ada, tp_qda]
ax.bar(algoritmos, tps)
ax.set_title('Comparação de Desempenho em tempo de Previsão (tempo de execução)')
plt.show()


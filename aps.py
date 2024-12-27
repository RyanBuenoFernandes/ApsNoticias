import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
from nltk.stem.lancaster import LancasterStemmer
import socket
import json
import xml.etree.ElementTree as ET
import mplcursors

# Carregar o JSON com as palavras-chaves
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Preparar dados de palavras e classes em listas
words = []  # Palavras do JSON
classes = []  # Classes (tags) do JSON
documentos = []  # Combinação das classes com cada palavra

stemmer = LancasterStemmer()  # Objeto responsavel pelo stemmer ( raiz das palavras)


# Função para preparar palavras (tokenização, remoção de stopwords, stemmer)
def preparar_palavras_JSON(words):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    words = [stemmer.stem(w.lower()) for w in words if w not in stopwords]
    words = sorted(list(set(words)))  # Remove palavras duplicadas e ordena
    return words


# coletar padrões e classes
def coletar_palavras_e_classes(intents_list):
    for intent in intents_list:
        # Coletar palavras e classe da tag principal
        for pattern in intent['patterns']:
            words.extend(nltk.word_tokenize(pattern.lower(), language='portuguese'))
            documentos.append((pattern.lower(), intent['tag']))


# Chama a função para coletar as palavras com sua respectiva classe
coletar_palavras_e_classes(intents['intents'])

# Preparar palavras únicas e classes únicas
words = preparar_palavras_JSON(words)
classes = sorted(list(set([doc[1] for doc in documentos])))

# Preparar dados de treinamento
treinamento = []  # Lista para armazenar os dados de treinamento
output_empty = [0] * len(classes)  # Lista de saída dos sentimentos

for doc in documentos:
    palavras_padrao = nltk.word_tokenize(doc[0], language='portuguese')
    palavras_padrao = [stemmer.stem(word.lower()) for word in palavras_padrao]

    # Criar vetor de palavras (bolsa de palavras) com base nas palavras únicas
    bolsa = np.zeros(len(words), dtype=int)
    for w in palavras_padrao:
        if w in words:
            bolsa[words.index(w)] = 1

    # Criar vetor de classes
    linha_de_saida = list(output_empty)
    linha_de_saida[classes.index(doc[1])] = 1

    treinamento.append((bolsa, linha_de_saida))

# Embaralhar dados de treinamento
random.shuffle(treinamento)

# Separar dados de entrada (X) e saída (Y)
treinar_x = np.array([linha[0] for linha in treinamento])  # Entradas (bolsa de palavras)
treinar_y = np.array([linha[1] for linha in treinamento])  # Saídas (classes)

# Criação da rede neural e treinamento
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(treinar_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(treinar_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(treinar_x, treinar_y, n_epoch=1000, batch_size=8, show_metric=True)  # 8 lotes de dados rodando 1000 vezes
model.save('model.tflearn')

print('Rede Neural Treinada!')


# Função para preparar os dados de entrada para a previsão das noticias
def preparar_dados_de_entrada(title, text):
    # Tokenização e pré-processamento do texto
    words_title = nltk.word_tokenize(title.lower())
    words_text = nltk.word_tokenize(text.lower())
    palavras_preprocessadas = preparar_palavras_JSON(words_title + words_text)

    # Criar o vetor de palavras (bolsa de palavras) com o mesmo comprimento que treinar_x[0]
    bolsa_de_palavras = np.zeros(len(words), dtype=int)
    for w in palavras_preprocessadas:
        if w in words:
            index = words.index(w)
            bolsa_de_palavras[index] = 1  # Marcar a presença da palavra na bolsa de palavras

    return bolsa_de_palavras


# Durante a análise das notícias, obtenha a probabilidade de cada categoria de acordo com título e o corpo da notícia
def analisar_sentimento_noticias(title, text):
    # Preparar dados de entrada
    bolsa_de_palavras = preparar_dados_de_entrada(title, text)

    # Alimentar as palavras na rede neural e obter a previsão
    previsoes = model.predict([bolsa_de_palavras])[0]

    # Identificar a classe com maior probabilidade
    predicted_class_index = np.argmax(previsoes)
    predicted_class = classes[predicted_class_index]

    # Exibir as probabilidades
    probabilidade_negativo = previsoes[classes.index("negativo")]
    probabilidade_positiva = previsoes[classes.index("positivo")]

    return predicted_class, probabilidade_negativo, probabilidade_positiva


# Lista para armazenar o sentimento de cada notícia
sentimentos_noticias = []


# Função para extrair título, corpo da notícia, data, URL
def carregar_noticias_do_xml(xml_file_path):
    tree = ET.parse(xml_file_path)  # TKinter cria a interface gráfica no formato de uma 'arvore', no qual, cada nó
    root = tree.getroot()  # representa uma noticia com suas informações particulares
    for noticia in root.findall('noticia'):
        titulo = noticia.find('titulo').text
        corpo = noticia.find('corpo').text
        link = noticia.find('link').text
        data = noticia.find('data').text
        # Extrair IP do link da notícia com a função
        ip = extrair_ip(link)
        # Obter o protocolo de rede com a função
        network_protocol = obter_protocolo_de_rede(ip)
        # Inserir na árvore (Treeview)
        treeview.insert('', 'end', text=titulo, values=(link, network_protocol, ip, data))


# Analisa o sentimento de todas as notícias
def obter_e_analisar_noticias_xml(xml_file_path):
    global sentimentos_noticias  # Usar a variável global sentimentos_noticias
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for noticia in root.findall('noticia'):
        titulo = noticia.find('titulo').text
        corpo = noticia.find('corpo').text
        data = noticia.find('data').text
        # analise cada noticia e joga dentro da lista sentimentos_noticias
        sentimento, probabilidade_negativo, probabilidade_positiva = analisar_sentimento_noticias(titulo, corpo)
        if sentimento is not None:
            sentimentos_noticias.append(
                (sentimento, float(probabilidade_negativo), float(probabilidade_positiva), data))
            print("Título da notícia processada:", titulo)


# Analisa o sentimento de uma única notícia
def analisar_sentimentos_noticicas(title, text):
    sentimento, probabilidade_negativo, probabilidade_positiva = analisar_sentimento_noticias(title, text)
    if sentimento is not None:
        # Plotar o gráfico de uma única notícia
        grafico_de_sentimento(title, probabilidade_negativo, probabilidade_positiva)


# Função para analisar o tipo de protocolo de rede de cada site
def obter_protocolo_de_rede(url):
    try:
        # Tentar estabelecer uma conexão TCP
        # biblioteca socket -----indicação do RICARDO-----
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(2)  # Tempo limite de 2 segundos
        tcp_socket.connect((url, 80))
        tcp_socket.close()
        return "TCP"  # Se a conexão TCP for bem-sucedida, o protocolo é TCP
    except Exception as tcp_error:
        try:
            # Tentar estabelecer uma conexão UDP
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.settimeout(2)  # Tempo limite de 2 segundos
            udp_socket.connect((url, 53))
            udp_socket.close()
            return "UDP"  # Se a conexão UDP for bem-sucedida, o protocolo é UDP
        except Exception as udp_error:
            return "Desconhecido"  # Se ambas as conexões falharem, o protocolo é desconhecido


# Função para extrair IP do link
def extrair_ip(link):
    hostname = link.split('//')[-1].split('/')[0]  # Extrair hostname
    try:
        ip = socket.gethostbyname(hostname)  # Obter o IP do hostname
    except Exception as e:
        ip = 'Erro de conexão'  # Em caso de falha de conexão ou erro
    return ip


# Função para lidar com o duplo clique em um título de notícia
def duplo_clique_noticia(event):
    item = treeview.item(treeview.selection())  # Obter o nó selecionado na árvore
    title = item['text']

    # Encontrar a notícia correspondente ao título no arquivo XML
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for noticia in root.findall('noticia'):
        if noticia.find('titulo').text == title:
            body = noticia.find('corpo').text
            analisar_sentimentos_noticicas(title, body)
            break  # Parar de iterar assim que a notícia correspondente for encontrada


def criar_grafico_geral():
    global sentimentos_noticias
    if sentimentos_noticias:
        # Calcular a média dos sentimentos negativos e positivos para obter a porcentagem
        total_negative = sum(sentimento[1] for sentimento in sentimentos_noticias)
        total_positive = sum(sentimento[2] for sentimento in sentimentos_noticias)

        avg_negative = total_negative / len(sentimentos_noticias)
        avg_positive = total_positive / len(sentimentos_noticias)

        porcentagem_negative = avg_negative * 100
        porcentagem_positive = avg_positive * 100

        labels = ['Negativo', 'Positivo']
        heights = [avg_negative, avg_positive]
        porcentagens = [porcentagem_negative, porcentagem_positive]
        colors = ['#ff0000', '#00ff00']
        plt.bar(labels, heights, color=colors)
        plt.xlabel('Sentimento')
        plt.ylabel('Probabilidade Média')
        plt.title('Análise de Sentimento de Notícias (Média)')

        # Adicionar anotações com as porcentagens ao lado das barras
        for i, (label, height, porcentagem) in enumerate(zip(labels, heights, porcentagens)):
            plt.text(i, height, f'{porcentagem:.2f}%', ha='center', va='bottom', color='black', fontsize=12)

        plt.get_current_fig_manager().window.state('zoomed')  # Maximizar a janela
        plt.show()


def calcular_media_por_dia(sentimentos_noticias):
    media_por_dia = []
    num_noticias_por_dia = 10
    datas_por_dia = []  # Lista para armazenar datas por dia
    for i in range(0, len(sentimentos_noticias), num_noticias_por_dia):
        noticias_do_dia = sentimentos_noticias[i:i + num_noticias_por_dia]
        datas_do_dia = [noticia[3] for noticia in noticias_do_dia]  # Extrair datas do dia
        datas_por_dia.append(datas_do_dia)

        total_negative = sum(sentimento[1] for sentimento in noticias_do_dia)
        total_positive = sum(sentimento[2] for sentimento in noticias_do_dia)

        media_negative = total_negative / num_noticias_por_dia
        media_positive = total_positive / num_noticias_por_dia
        total_noticias = len(noticias_do_dia)

        porcent_negativo = total_negative / total_noticias
        porcent_positivo = total_positive / total_noticias

        porcentagem_negative = porcent_negativo * 100
        porcentagem_positive = porcent_positivo * 100
        media_por_dia.append((media_negative, media_positive, porcentagem_negative, porcentagem_positive))
    return media_por_dia, datas_por_dia  # Retorna também a lista de datas por dia


def criar_grafico_por_dia():
    global sentimentos_noticias
    media_por_dia, datas_por_dia = calcular_media_por_dia(sentimentos_noticias)
    # Extrair as datas para os dias correspondentes
    dias = ['Dia ' + str(i) for i in range(1, len(media_por_dia) + 1)]  # Lista de datas formatadas como string
    datas = [' // '.join(datas) for datas in datas_por_dia]

    media_negativa = [media[0] for media in media_por_dia]
    media_positiva = [media[1] for media in media_por_dia]
    porcentagens_negativas = [media[2] for media in media_por_dia]
    porcentagens_positivas = [media[3] for media in media_por_dia]

    largura_barra = 0.35  # largura das barras
    x = np.arange(len(dias))  # posição dos dias no eixo x

    fig, ax = plt.subplots()
    barra_negativa = ax.bar(x - largura_barra / 2, media_negativa, largura_barra, label='Negativo', color='#ff0000')
    barra_positiva = ax.bar(x + largura_barra / 2, media_positiva, largura_barra, label='Positivo', color='#00ff00')

    ax.set_xlabel('Data das Notícias')  # Atualizar o rótulo do eixo x
    ax.set_ylabel('Probabilidade Média')
    ax.set_title('Análise de Sentimento de Notícias por Dia')
    ax.set_xticks(x)
    ax.set_xticklabels(dias, rotation=45, ha='right')  # Rotação das datas para melhor visualização
    ax.legend()

    for i, (barra_neg, barra_pos, porc_neg, porc_pos) in enumerate(
            zip(barra_negativa, barra_positiva, porcentagens_negativas, porcentagens_positivas)):
        ax.text(barra_neg.get_x() + barra_neg.get_width() / 2., media_negativa[i], f'{porc_neg:.2f}%', ha='center',
                va='bottom')
        ax.text(barra_pos.get_x() + barra_pos.get_width() / 2., media_positiva[i], f'{porc_pos:.2f}%', ha='center',
                va='bottom')

    # Adicionar anotações interativas com as datas
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(datas[sel.target.index]))
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


# Função para plotar o gráfico de análise de sentimento de cada noticia
def grafico_de_sentimento(title, probabilidade_negativo, probabilidade_positiva):
    labels = ['Negativo', 'Positivo']
    heights = [float(probabilidade_negativo), float(probabilidade_positiva)]
    colors = ['#ff0000', '#00ff00']
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, heights, color=colors)
    plt.bar(labels, heights, color=colors)
    plt.xlabel('Sentimento')
    plt.ylabel('Probabilidade')
    plt.title(title)
    # Adicionar anotações com as porcentagens ao lado das barras
    for i, (label, height) in enumerate(zip(labels, heights)):
        plt.text(i, height, f'{height:.2f}%', ha='center', va='bottom', color='black', fontsize=12)
    plt.get_current_fig_manager().window.state('zoomed')  # Maximizar a janela do gráfico
    plt.show()


# Criar a interface gráfica
root = tk.Tk()
root.title("Análise de Sentimento de Notícias")

# Criar a árvore para exibir as notícias, agora com uma nova coluna para os protocolos
treeview = ttk.Treeview(root, columns=('Link', 'Protocolo', 'IP', 'Data'))
treeview.heading('#0', text='Título')
treeview.heading('Link', text='Link')
(treeview.heading('Protocolo', text='Protocolo'))  # Novo cabeçalho para a coluna de protocolo
treeview.heading('IP', text='IP')
treeview.heading('Data', text='Data')
treeview.bind('<Double-1>', duplo_clique_noticia)  # Bind duplo clique para chamar a função duplo_clique_noticia
treeview.pack(expand=True, fill='both')
# Criar uma aba separada para exibir o gráfico combinado
tab_control = ttk.Notebook(root)
tab_sentiment = ttk.Frame(tab_control)
tab_control.add(tab_sentiment, text='Análise de Sentimento')
tab_control.place(x=680, y=700, width=200, height=65)

# Definir as dimensões da janela para tela cheia
largura_tela = root.winfo_screenwidth()
altura_tela = root.winfo_screenheight()
root.geometry(f"{largura_tela}x{altura_tela}")

# Carregar as notícias do arquivo XML
xml_file_path = "noticias_analisadas.xml"
carregar_noticias_do_xml(xml_file_path)
button_combined_graph = tk.Button(tab_sentiment, text='Gráfico Geral', command=criar_grafico_geral)
button_combined_graph.pack()
button_combined_graph.config(bg='#6495ED')  # Mudar a cor do botão

# Crie um botão para chamar a função criar_grafico_por_dia
button_daily_average_graph = tk.Button(tab_sentiment, text='Gráfico de Dias', command=criar_grafico_por_dia, height=15)
button_daily_average_graph.pack()
button_daily_average_graph.config(bg='#FFD700')
obter_e_analisar_noticias_xml(xml_file_path)
root.mainloop()

#teste
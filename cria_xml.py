from bs4 import BeautifulSoup
import feedparser
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import requests
from nltk.stem.lancaster import LancasterStemmer
import socket
import json
import xml.etree.ElementTree as ET

stemmer = LancasterStemmer()

# Leitura do JSON
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)


# Preparação dos dados
def prepare_words(words):
    if isinstance(words, str):
        words = nltk.word_tokenize(words, language='portuguese')

    stopwords = nltk.corpus.stopwords.words('portuguese')
    words = [stemmer.stem(w.lower()) for w in words if w not in stopwords]
    words = sorted(list(set(words)))
    return words


words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern.lower(), language='portuguese')
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = prepare_words(words)
classes = sorted(list(set(classes)))

# Preparação dos dados de treinamento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    bag = np.zeros(len(words), dtype=int)
    for w in pattern_words:
        if w in words:
            bag[words.index(w)] = 1
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append((bag, output_row))
random.shuffle(training)
training_data = np.array(training, dtype=object)
train_x = list(training_data[:, 0])
train_y = list(training_data[:, 1])

# Criação da rede neural e treinamento
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

print('Rede Neural Treinada!')


# Função para obter o conteúdo HTML de uma URL
def get_page_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None


# Função para extrair o texto de uma página HTML
def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Extrair texto do HTML, excluindo tags e scripts
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text()


# Função para preparar os dados de entrada para a previsão
def prepare_input_data(title, text):
    # Tokenização e pré-processamento do texto
    words_title = nltk.word_tokenize(title.lower())
    words_text = nltk.word_tokenize(text.lower())
    preprocessed_words = prepare_words(words_title + words_text)

    # Criar o vetor de palavras (bag of words) com o mesmo comprimento que train_x[0]
    bag_of_words = np.zeros(len(words), dtype=int)
    for w in preprocessed_words:
        if w in words:
            index = words.index(w)
            if index < len(train_x[0]):  # Verifica se o índice está dentro do limite do vetor
                bag_of_words[index] = 1

    return bag_of_words


# Durante a análise das notícias, obtenha a probabilidade de cada categoria
def analyze_news_sentiment(url):
    html_content = get_page_content(url)
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string if soup.title else ""
        text_content = extract_text_from_html(html_content)
        # Preparar dados de entrada
        bag_of_words = prepare_input_data(title, text_content)

        # Alimentar as palavras na rede neural e obter a previsão
        predictions = model.predict([bag_of_words])[0]

        # Identificar a classe com maior probabilidade
        predicted_class_index = np.argmax(predictions)
        predicted_class = classes[predicted_class_index]

        # Exibir as probabilidades formatadas
        prob_negative = "{:.12f}".format(predictions[classes.index("negativo")])
        prob_positive = "{:.12f}".format(predictions[classes.index("positivo")])

        return predicted_class, prob_negative, prob_positive
    else:
        return None, None, None


# Lista para armazenar o sentimento de cada notícia
news_sentiments = []


def get_and_analyze_news_from_xml(xml_file_path, output_xml_file_path):
    global news_sentiments  # Usar a variável global `news_sentiments`

    # Criar o elemento raiz do XML
    root = ET.Element("noticias")

    # Parsear o arquivo XML de entrada
    feed = feedparser.parse(xml_file_path)

    # Lista de URLs das notícias que você não deseja incluir no arquivo XML
    urls_indesejados = ["https://g1.globo.com/economia/agronegocios/noticia/2024/03/21/pitaya-tem-sabor-e-a-especie-mais-doce-a-baby-do-cerrado-e-nativa-do-brasil.ghtml",
                        "https://www.cnnbrasil.com.br/nacional/desmatamento-na-mata-atlantica-cai-59-entre-janeiro-e-agosto-de-2023/",
                        "https://www.cnnbrasil.com.br/nacional/duas-oncas-pintadas-sao-atropeladas-no-interior-de-sp-em-uma-semana-especie-esta-ameacada-2/",
                        "https://www.cnnbrasil.com.br/nacional/amazonia-tem-2o-trimestre-com-maior-desmate-desde-2016/",
                        "https://www.cnnbrasil.com.br/nacional/ameacado-de-extincao-lobo-guara-e-flagrado-em-marica-assista/",
                        "https://www.cnnbrasil.com.br/nacional/focos-de-incendio-aumentaram-cerca-de-283-entre-janeiro-e-julho-no-brasil/",
                        "https://www.cnnbrasil.com.br/nacional/em-35-anos-amazonia-perdeu-vegetacao-equivalente-ao-territorio-do-chile/",
                        "https://www.cnnbrasil.com.br/nacional/brasil-esta-em-uma-trajetoria-errada-diz-especialista-sobre-emissoes/",
                        "https://www.bbc.com/portuguese/articles/crgrywl9ypxo",
                        "https://www.bbc.com/portuguese/articles/ckdpygjg99go",
                        "https://www.bbc.com/portuguese/brasil-62718299",
                        "https://www.bbc.com/portuguese/internacional-55168713",
                        "https://www.bbc.com/portuguese/brasil-54221704",
                        "https://www.bbc.com/portuguese/brasil-54213503",
                        "https://g1.globo.com/df/distrito-federal/noticia/2024/04/03/pesquisadores-da-unb-criam-cerveja-com-polen-de-abelha-e-seriguela-voce-provaria.ghtml",
                        "https://g1.globo.com/go/goias/noticia/2024/03/24/sapo-foguetinho-conheca-anfibio-de-dois-centimetros-que-pode-virar-simbolo-do-cerrado.ghtml",
                        "https://www.cnnbrasil.com.br/nacional/quase-mil-hectares-foram-queimados-em-fevereiro-em-todo-o-brasil-aponta-ipam/",
                        "https://g1.globo.com/podcast/resumao-diario/noticia/2024/04/12/resumao-diario-do-jn-stf-forma-maioria-para-ampliar-foro-privilegiado-de-deputados-e-senadores-e-cai-o-desmatamento-da-amazonia-mas-cresce-a-devastacao-do-cerrado.ghtml"]

    for entry in feed.entries:
        if entry.link not in urls_indesejados:
            # Processar a notícia apenas se o URL não estiver na lista de URLs indesejados
            sentiment, prob_negative, prob_positive = analyze_news_sentiment(entry.link)
            if sentiment is not None:
                # Criar elementos para cada notícia
                news_element = ET.SubElement(root, "noticia")
                title_element = ET.SubElement(news_element, "titulo")
                title_element.text = entry.title
                link_element = ET.SubElement(news_element, "link")
                link_element.text = entry.link
                body_element = ET.SubElement(news_element, "corpo")
                body_element.text = extract_text_from_html(get_page_content(entry.link))
                date_element = ET.SubElement(news_element, "data")  # Adicionar elemento para a data
                date_element.text = entry.published  # Inserir a data de publicação
                # Adicionar notícia ao XML
                news_sentiments.append((sentiment, float(prob_negative), float(prob_positive)))
                print("Título da notícia processada:", entry.title)
                print("Link da notícia:", entry.link)

        else:
            print("Notícia ignorada:",
                  entry.title)  # Se o URL estiver na lista, imprimir uma mensagem de notícia ignorada

    # Criar o objeto do ElementTree
    tree = ET.ElementTree(root)

    # Escrever o XML para o arquivo de saída
    tree.write(output_xml_file_path, encoding="utf-8", xml_declaration=True)


# Função para analisar o IP de uma lista de links de notícias
def analyze_ip_from_xml(xml_file_path):
    # Analisar o arquivo XML para obter as notícias
    feed = feedparser.parse(xml_file_path)
    for entry in feed.entries:
        link = entry.link
        protocol = get_protocol_for_site(link)
        if protocol:
            print("Protocolo:", protocol)
        # Obter e exibir o IP do site
        ip = get_ip_for_site(link)
        if ip:
            print("Link:", link)
            print("IP:", ip)
            print()


# Função para obter o IP de um site
def get_ip_for_site(url):
    try:
        hostname = url.split("//")[-1].split("/")[0]  # Extrair o nome do host do URL
        ip = socket.gethostbyname(hostname)  # Obter o endereço IP do host
        return ip
    except Exception as e:
        print("Erro ao obter o IP:", e)
        return None


# Função para obter o protocolo de um link
def get_protocol_for_site(url):
    try:
        protocol = url.split(":")[0]  # Extrair o protocolo do URL
        return protocol
    except Exception as e:
        print("Erro ao obter o protocolo:", e)
        return None


# Carregar as notícias do arquivo XML
xml_file_path = r"C:\Users\buesb\OneDrive\Documentos\aps\COLOCAR_O_NOME_AQUI.xml"
# Chamar a função para analisar o protocolo das notícias
analyze_ip_from_xml(xml_file_path)

# Define feed como uma variável global ou em um escopo acessível
feed = feedparser.parse(xml_file_path)
output_xml_file_path = "dia1.xml"
get_and_analyze_news_from_xml(xml_file_path, output_xml_file_path)
print("Arquivo de saída salvo em:", output_xml_file_path)
# Iniciando teste aqui
print("Lista de sentimentos das noticias:")
for sentiment_data in news_sentiments:
    print(sentiment_data)

# TENTANDO TIRAR AS COISAS

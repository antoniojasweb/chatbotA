# -------------------------------------------------------------------
#pip install streamlit pandas faiss-cpu sentence-transformers requests
#pip install openai langchain PyPDF2 langdetect langchain-community sckit-learn
#pip install spacy scikit-learn openpyxl pymupdf gtts
#pip install pymupdf rank_bm25
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Importar las librerías necesarias
# -------------------------------------------------------------------
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import os
import requests
#import time

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

# Para la conversión de texto a voz
from gtts import gTTS
import io  # Necesario para manejar el buffer de audio
import base64 # Necesario para incrustar audio directamente en HTML

# Para grabar audio desde el micrófono
from scipy.io.wavfile import write
import speech_recognition as sr
import tempfile
from streamlit_realtime_audio_recorder import audio_recorder
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
# -------------------------------------------------------------------

# David
import pandas as pd
import fitz  # PyMuPDF
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
#from sentence_transformers import SentenceTransformer
#import torch
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import re
from typing import List, Dict, Tuple

import nltk

# -------------------------------------------------------------------
# --- Definición de rutas y ficheros de datos ---
url = "https://raw.githubusercontent.com/antoniojasweb/chatbotN/main/pdf/"
FilePDF = "25_26_OFERTA_por_Familias.pdf"
FileExcel = "oferta_formativa_completa.xlsx"
FileLogo = "Logo.png"
TituloAPP = "Ciclos Formativos en Extremadura: 25/26"
FileDistancias = "distanciasJD.xlsx"  # Fichero de distancias entre institutos y municipios

# --- Modelos ---
ModeloEmbeddings = 'paraphrase-multilingual-MiniLM-L12-v2'
# Otro modelo para Embeddings, de Sentence Transformers, para Ingés: 'all-MiniLM-L6-v2'
ModeloEvaluacion = 'all-mpnet-base-v2'  # Modelo para evaluación de respuestas
# Otros posibles modelos de Evaluación: all-MiniLM-L6-v2, sentence-t5-base, e5-large-v2, etc

# --- Configuración de la API de Gemini ---
# Otro modelo posible: imagen-3.0-generate-002
Model_Gemini = "gemini-2.0-flash"
API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# --- Definición de funciones ---
# -------------------------------------------------------------------
def color_to_rgb(color_int):
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return (r, g, b)

# Descargar el fichero del logo del chatbot
def descargar_logo(fichero_logo):
    if not os.path.exists(FileLogo):
        FileURL = url + fichero_logo
        if not os.path.exists(fichero_logo):
            response = requests.get(FileURL)
            # Verificamos que la solicitud fue exitosa
            if response.status_code == 200:
                # Abrimos el archivo en modo de lectura binaria
                with open(FileLogo, 'wb') as file:
                    file.write(response.content)
                print("Logo descargado: " + fichero_logo)
            else:
                print(f"Error al descargar el logo: {response.status_code}")

# Descargar fichero PDF desde URL
def descargar_pdf(fichero_pdf):
    if not os.path.exists(FilePDF):
        FileURL = url + fichero_pdf
        if not os.path.exists(fichero_pdf):
            response = requests.get(FileURL)
            # Verificamos que la solicitud fue exitosa
            if response.status_code == 200:
                # Abrimos el archivo PDF en modo de lectura binaria
                with open(FilePDF, 'wb') as file:
                    file.write(response.content)
                print("Fichero PDF: " + fichero_pdf + " descargado.")
            else:
                print(f"Error al descargar el PDF: {response.status_code}")

# Extraer información del PDF y guardarla en un DataFrame
def extraer_informacion_pdf(fichero_pdf):
    # Comprobar si el archivo Excel ya existe, y lo eliminamos, para generar uno nuevo
    if os.path.exists(FileExcel):
        os.remove(FileExcel)

    # Abrir el PDF
    #print("Fichero PDF procesado: " + fichero_pdf)
    doc = fitz.open(fichero_pdf)

    # Inicializar una lista para almacenar los datos
    data = []
    familia_actual = ""
    grado_actual = ""
    codigo_ciclo = ""
    nombre_ciclo = ""
    provincia_actual = ""
    turno = "Diurno"  # Valor por defecto
    instituto = ""
    municipio = ""
    bilingue = ""
    nuevo = ""

    # Recorrer cada página y extraer información
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                for span in l["spans"]:
                    text = span["text"].strip()
                    rgb = color_to_rgb(span["color"])
                    font = span["font"]
                    is_bold = "Bold" in font or "bold" in font.lower()
                    bilingue = ""
                    nuevo = ""

                    # Familia profesional (verde y mayúsculas)
                    if text.isupper() and rgb[1] > 120 and rgb[0] < 100 and rgb[2] < 100:
                        familia_actual = text

                    # Grado formativo (naranja y mayúsculas)
                    # elif text.isupper() and rgb[0] > 150 and rgb[1] > 90 and rgb[2] < 50:
                    #     grado_actual = text
                    elif rgb[0] > 150 and 90 < rgb[1] < 190 and rgb[2] > 80:
                        grado_actual = text  # Guardamos el texto literal como grado actual

                    # Ciclo formativo (azul, con código entre paréntesis)
                    elif text.startswith("(") and ")" in text and rgb[2] > 100:
                        if ")" in text:
                            codigo_ciclo = text.split(")")[0].strip("()")
                            nombre_ciclo = text.split(")", 1)[1].strip()

                    # Provincia (negrita + negro)
                    elif text in ["BADAJOZ", "CÁCERES"] and is_bold:
                        provincia_actual = text

                    # Centro educativo (normal, negro, contiene ' - ' al menos 2 veces)
                    elif text.count(" - ") >= 2 and not is_bold and rgb == (0, 0, 0):
                        try:
                            municipio, instituto, curso_raw = text.split(" - ")

                            if "acreditado" in text:
                                instituto += " (Acreditado en Calidad)"

                            if "1ºC" in text:
                                curso = "1ºC"
                            elif "2ºC" in text:
                                curso = "2ºC"

                        except ValueError:
                            continue  # línea malformada

                    elif text in ['Vespertino','Diurno','Bilingüe', 'Bilingue', 'Nuevo']:
                        tipo = 'Viejo'
                        if 'Vespertino' in text:
                            turno = 'Vespertino'
                        elif 'Diurno' in text:
                            turno = 'Diurno'
                        #elif text in ['Bilingüe', 'Bilingue']:
                        #    bilingue = 'bilingüe'
                        elif 'Nuevo' in text:
                            tipo = 'Nuevo'

                        # Añadir fila: Sólo si el curso es de 1ºC
                        if "1ºC" in curso:
                            data.append({
                                "Familia": familia_actual.title().strip(),
                                #"Codigo": codigo_ciclo.strip(),
                                "Ciclo": nombre_ciclo.title().strip(),
                                "Grado": grado_actual.strip(),
                                "Instituto": instituto.strip(),
                                "Municipio": municipio.title().strip(),
                                "Provincia": provincia_actual.title().strip(),
                                "Turno": turno,
                                #"Idiomas": bilingue,
                                "Tipo": tipo
                            })

    if data:
        # Convertir a DataFrame
        df = pd.DataFrame(data)

        # ordenar por la columna Familia, Instituto y Ciclo, en orden ascendente
        df.fillna("", inplace=True)  # Reemplaza valores NaN por una cadena vacía
        df.sort_values(['Familia', 'Instituto', 'Ciclo'], ascending=[True, True, True], inplace=True)

        # Exportar a Excel
        df.to_excel(FileExcel, index=False)
        print("Fichero Excel creado : " + FileExcel)
    else:
        print("No se encontraron datos válidos en el PDF.")
        # df = pd.DataFrame(columns=[
        #     "Familia", "Ciclo", "Grado", "Instituto", "Municipio", "Provincia", "Turno", "Tipo"
        # ])

    # Si quieres mostrar el DataFrame en Streamlit, puedes descomentar la siguiente línea
    #st.write(df)

    # Mostrar primeras filas
    #print(df.head())
    #df.head()

    #return df

# Descargar fichero Distancias desde URL
def descargar_distancias(FicheroDistancias):
    if not os.path.exists(FicheroDistancias):
        FileURL = url + FicheroDistancias
        if not os.path.exists(FicheroDistancias):
            response = requests.get(FileURL)
            # Verificamos que la solicitud fue exitosa
            if response.status_code == 200:
                # Abrimos el archivo PDF en modo de lectura binaria
                with open(FicheroDistancias, 'wb') as file:
                    file.write(response.content)
                print("Fichero Distancias: " + FicheroDistancias + " descargado.")
            else:
                print(f"Error al descargar el fichero Distancias: {response.status_code}")

    #return pd.read_excel(FicheroDistancias)

#--------------------------------------------------------------------

class TextPreprocessor:
    """
    Clase para preprocesar texto en español.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('spanish'))

    def preprocess(self, text: str) -> str:
        """Preprocesa un texto en español."""
        # Convertir a minúsculas
        text = text.lower()

        # Tokenizar usando NLTK
        tokens = word_tokenize(text)

        # Eliminar stopwords usando NLTK
        stop_words = set(stopwords.words('spanish'))
        tokens = [t for t in tokens if t not in stop_words]

        return ' '.join(tokens)

class SparseRetriever:
    """
    Implementa búsqueda dispersa usando BM25.
    """
    def __init__(self, documentos: List[str]):
        self.preprocessor = TextPreprocessor()

        # Preprocesar documentos
        processed_docs = [self.preprocessor.preprocess(doc) for doc in documentos]

        # Tokenizar para BM25
        tokenized_docs = [doc.split() for doc in processed_docs]

        # Inicializar BM25
        self.bm25 = BM25Okapi(tokenized_docs)

        # Guardar documentos originales
        self.documentos = documentos

    def buscar(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Realiza búsqueda BM25."""
        # Preprocesar query
        processed_query = self.preprocessor.preprocess(query)
        query_tokens = processed_query.split()

        # Obtener scores BM25
        scores = self.bm25.get_scores(query_tokens)

        # Obtener top_k resultados
        if xDistancias:
            top_indices = np.argsort(scores)[::-1][:top_k]
        else:
            top_indices = np.argsort(scores)[::-1]

        # Preparar resultados
        resultados = []
        resultados = [(idx, scores[idx]) for idx in top_indices]

        return resultados

class DenseRetriever:
    """
    Implementa búsqueda densa usando embeddings.
    """
    def __init__(self, documentos: List[str]):
        # Cargar modelo de embeddings multilingüe
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

        # Generar y almacenar embeddings
        self.embeddings = self.model.encode(documentos, show_progress_bar=True)

        # Guardar documentos originales
        self.documentos = documentos

    def buscar(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Realiza búsqueda por similitud de embeddings."""
        # Generar embedding de la query
        query_embedding = self.model.encode([query])

        # Calcular similitud
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Obtener top_k resultados
        if xDistancias:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            top_indices = np.argsort(similarities)[::-1]

        # Preparar resultados
        resultados = []
        resultados = [(idx, similarities[idx]) for idx in top_indices]

        return resultados

class HybridRetriever:
    """
    Combina resultados de búsqueda dispersa y densa.
    """
    def __init__(self, documentos: List[str],
                 weight_sparse: float = 0.3,
                 weight_dense: float = 0.7):
        self.sparse_retriever = SparseRetriever(documentos)
        self.dense_retriever = DenseRetriever(documentos)
        self.weight_sparse = weight_sparse
        self.weight_dense = weight_dense
        self.documentos = documentos

    def buscar(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Realiza búsqueda híbrida y fusiona resultados.
        """
        # Obtener resultados de ambos retrievers
        sparse_results = self.sparse_retriever.buscar(query, top_k=top_k)
        dense_results = self.dense_retriever.buscar(query, top_k=top_k)

        # Combinar scores
        combined_scores = {}
        for idx, score in sparse_results:
            combined_scores[idx] = score * self.weight_sparse

        for idx, score in dense_results:
            if idx in combined_scores:
                combined_scores[idx] += score * self.weight_dense
            else:
                combined_scores[idx] = score * self.weight_dense

        # Ordenar resultados finales
        if xDistancias:
            sorted_results = sorted(combined_scores.items(),
                              key=lambda x: x[1],
                              reverse=True)[:top_k]
        else:
            sorted_results = sorted(combined_scores.items(),
                              key=lambda x: x[1],
                              reverse=True)

        # Preparar resultados
        resultados = []
        for idx, score in sorted_results:
            resultados.append({
                'Familia': df.iloc[idx]['Familia'],
                'Ciclo': df.iloc[idx]['Ciclo'],
                'Grado': df.iloc[idx]['Grado'],
                'Instituto': df.iloc[idx]['Instituto'],
                'Municipio': df.iloc[idx]['Municipio'],
                'Provincia': df.iloc[idx]['Provincia'],
                'Turno': df.iloc[idx]['Turno'],
                'Tipo': df.iloc[idx]['Tipo'],
                'Score': score
            })

        return resultados

# -------------------------------------------------------------------
# FileHybrida = "Hybrida.xlsx"
def busqueda_hibrida(query: str, origen:str, top_k: int = 10):
    """
    Realiza una búsqueda híbrida utilizando el retriever híbrido.
    """
    # results = hybrid_retriever.buscar(query, top_k=top_k)
    # return results
    # basada en combinación de tema y palabras clave
    #query = "edificación"
    # Habría que coger la población a través del Chat
    #origen = "Pueblonuevo del Guadiana"

    print("\nBúsqueda:", query)
    print("Procesando ...")
    hybrid_results = st.session_state.hybrid_retriever.buscar(query, top_k)
    # # Mostrar las primeras 5 líneas
    # for resultado in hybrid_results[:5]:
    #     print(resultado)

    df_resultados = pd.DataFrame(columns=["familia", "ciclo", "grado", "instituto", "municipio", "provincia", "score", "distancia"])
    #print(df_resultados)

    # df_resultados = pd.DataFrame(columns=[
    #     "Familia", "Ciclo", "Grado", "Instituto", "Municipio", "Provincia", "Turno", "Tipo", "Score", "Distancia"
    # ])

    resultados = []
    #Score = float(res.get('Score', 0.0))
    #municipio = res.get('Municipio', "Desconocido")
    for res in hybrid_results:
        distanciaKms = filtrar_distancia(distancias, origen, res['Municipio'])
        fila = {
            "Familia": res['Familia'],
            "Ciclo": res['Ciclo'],
            "Grado": res['Grado'],
            "Instituto": res['Instituto'],
            "Municipio": res['Municipio'],
            "Provincia": res['Provincia'],
            "Turno": res['Turno'],
            "Tipo": res['Tipo'],
            "Score": float(res['Score']),  # convertir np.float64 a float nativo (opcional pero recomendable)
            "Distancia": distanciaKms
        }
        resultados.append(fila)
        #print(fila)

    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.head())

    if xDistancias: # Ordenado por distancia
        df_ordenado = df_resultados.sort_values(by=["Distancia","Score"], ascending=[True, False])
        salida = df_ordenado.head(top_k)
    else:   # Ordenado por score
        df_ordenado = df_resultados.sort_values(by=["Score","Distancia"], ascending=[False, True])
        salida = df_ordenado

    # df_resultados.to_excel(FileHybrida, index=False)
    # print("Fichero Excel creado : " + FileHybrida)

    return salida

# -------------------------------------------------------------------

# --- Función para convertir texto a audio y obtenerlo en base64 ---
# def text_to_audio_base64(text, lang='es'):
#     """
#     Convierte texto a audio usando gTTS y devuelve el audio codificado en base64.
#     """
#     try:
#         text = text.replace('*', ' ')
#         tts = gTTS(text=text, lang=lang, slow=False)
#         audio_buffer = io.BytesIO()
#         tts.write_to_fp(audio_buffer)
#         audio_buffer.seek(0)

#         # Codificar el buffer de audio en base64
#         audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
#         # Limpiar el buffer
#         audio_buffer.close()
#         # Devolver el audio en formato base64
#         print("Audio generado y codificado en base64.")
#         return audio_base64
#     except Exception as e:
#         print(f"Error al generar audio: {e}")
#         return None

def grabar_audio():
    #st.write("Puedes grabar tu pregunta usando el micrófono.")
    # st.write("La grabación se inicia al pulsar el botón del micrófono, y finaliza al volver a pulsar el botón.")
    #col1, col2 = st.columns([5,10])
    #with col1:
    # Usar el componente de grabación de audio
    result = audio_recorder(
        interval=50,
        threshold=-60,
        silenceTimeout=200,
    )

    if result:
        if result.get('status') == 'stopped':
            audio_data = result.get('audioData')
            if audio_data:
                # Decodificar el audio base64 y mostrarlo
                print("Audio grabado correctamente.")
                audio_bytes = base64.b64decode(audio_data)
                audio_file = io.BytesIO(audio_bytes)
                # Mostrar el audio grabado
                #st.write(audio_file)
                #st.audio(audio_file, format="audio/webm")
            else:
                #pass
                st.error("Repite")
        elif result.get('error'):
                #st.error(f"Error: {result.get('error')}")
                st.error("Error")

        if audio_bytes is not None:
            # Reconocer el audio y convertirlo a texto
            recognizer = sr.Recognizer()
            temp_audio_file_path = None # Initialize to None

            try:
                audio_segment = AudioSegment.from_file(audio_file, format="webm")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    #temp_audio_file.write(audio_bytes)
                    temp_audio_file_path = temp_audio_file.name
                    audio_segment.export(temp_audio_file_path, format="wav")

                with sr.AudioFile(temp_audio_file_path) as source:
                    # Ajustar el umbral de energía del ruido para el reconocimiento
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio, language="es-ES")
                        #with col6:
                            # Mostrar la transcripción del audio
                            #st.write("🗣️ ", text)
                        user_query = text
                    except sr.UnknownValueError:
                        st.error("No se pudo entender el audio.")
                        user_query = None
                    except sr.RequestError as e:
                        st.error(f"Error al conectarse con el servicio de reconocimiento: {e}")
                        user_query = None
            finally:
                # Clean up the temporary file
                if temp_audio_file_path and os.path.exists(temp_audio_file_path):
                    os.remove(temp_audio_file_path)
    else:
        print("No se ha grabado audio. Puedes escribir tu consulta a continuación.")
        user_query = None

    return user_query

def lanzar_consulta(user_query):
    """
    Lanza la consulta al modelo RAG y muestra la respuesta.
    """
    #if st.session_state.excel_data is not None and st.session_state.faiss_index is not None:
    #st.write("Consultado1 ...")
    if st.session_state.hybrid_retriever is not None:
        #st.write("Consultado2 ...")
        # Reemplazar términos específicos en la consulta del usuario para mejorar la búsqueda
        #consulta_modificada = consulta_usuario.replace("Ciudad", "Municipio")
        #consulta_modificada = consulta_usuario.replace("Vespertino", "Tarde")
        equivalencias = {"Ciudad": "Municipio",
                        "Localidad": "Municipio",
                        "Tarde": "Vespertino",
                        "Nuevos": "Nuevo",
                        "Ciclo Formativo": "Ciclo",
                        "Centro Educativo":"Instituto"}

        if user_query:
            # Reemplazar términos en la consulta del usuario
            user_query = user_query.strip()  # Limpiar espacios al inicio y final

            # Reemplazar equivalencias en la consulta del usuario
            for clave, valor in equivalencias.items():
                user_query = user_query.replace(clave, valor)

            #Localidad = "Badajoz"  # Origen por defecto, se puede cambiar según la consulta
            top_k = 10  # Número de resultados a mostrar
            with st.spinner("Procesando ...", show_time=True):
                df_response = busqueda_hibrida(user_query, Localidad, top_k)

            #with st.spinner("Gemini ...", show_time=True):
                df_response['combined_text'] = df_response.apply(
                    lambda row: f"Ciclo: {row.get('Ciclo', '')}. Grado: {row.get('Grado', '')}. Familia: {row.get('Familia', '')}. Instituto: {row.get('Instituto', '')}. Municipio: {row.get('Municipio', '')}. Provincia: {row.get('Provincia', '')}. Turno: {row.get('Turno', '')}. Tipo: {row.get('Tipo', '')}. Score: {row.get('Score', '')}. Distancia: {row.get('Distancia', '')}",
                    axis=1
                )

                # Rellenar cualquier NaN con cadena vacía para evitar errores de embedding
                #df_response['combined_text'] = df_response['combined_text'].fillna('')
                corpus = df_response['combined_text'].tolist()
                context = "\n".join(corpus)

                #print(f"CONTEXTO: {context}")

                response = consultar_gemini(context, user_query)

                print(f"RESPUESTA: {response}")

            # Evaluar
            evaluar_respuesta(response, context, user_query)

            with st.chat_message("assistant"):
                st.write(response)
                # Muestra los documentos recuperados para depuración o información al usuario
                # Mensaje = "Ver información recuperada"
                # if xDistancias:
                #     Mensaje = "Ver información recuperada: " + str(top_k) + " opciones más relevantes"

                # with st.expander(Mensaje):
                #     retrieved_docs_df = pd.DataFrame(response[0])  # Convertir a DataFrame

                #     if retrieved_docs_df.empty:
                #         st.warning("El DataFrame está vacío.")
                #     else:
                #         st.write(retrieved_docs_df.head())
                #         st.dataframe(retrieved_docs_df, use_container_width=True)

                # Convertir respuesta del bot a audio base64
                #audio_b64 = text_to_audio_base64(response, lang='es')
                #if audio_b64:
                #    st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")

            # Añadir la respuesta al historial de chat
            st.session_state.chat_history.append({"role": "assistant", "content": response})

def mostrar_logo_titulo():
    """
    Muestra el logo del chatbot en la aplicación Streamlit.
    """
    col1, col2 = st.columns([1,2])
    with col1:
        if os.path.exists(FileLogo):
            image = Image.open(FileLogo)
            st.image(image, width=150)
    with col2:
        st.title(TituloAPP)

def inicializar_entorno():
    """
    Inicializa el entorno de la aplicación Streamlit.
    Configura las variables de sesión necesarias.
    """
    if 'model' not in st.session_state:
        st.session_state.model = None  # Modelo de embeddings
    if 'excel_data' not in st.session_state:
        st.session_state.excel_data = None  # Datos del DataFrame
    if 'excel_kms' not in st.session_state:
        st.session_state.excel_kms = None  # Datos del DataFrame

    #if 'faiss_index' not in st.session_state:
    #    st.session_state.faiss_index = None  # Índice FAISS
    if 'corpus' not in st.session_state:
        st.session_state.corpus = None  # Corpus de documentos
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Historial de chat
    if 'messages' not in st.session_state: # Mensajes del chat
        st.session_state.messages = []  # Mensajes del chat
    if 'hybrid_retriever' not in st.session_state:
        st.session_state.hybrid_retriever = None  # Retriever híbrido


def filtrar_distancia(tabla, origen, destino):
    #tabla = tabla.copy()
    if origen == "":
        origen = "Badajoz"
    tabla["origen"] = tabla["origen"].str.strip().str.lower()
    tabla["destino"] = tabla["destino"].str.strip().str.lower()

    tabla_filtrada = tabla.loc[(tabla["origen"] == origen.lower()) & (tabla["destino"] == destino.lower())]

    if not tabla_filtrada.empty:
        distanciaKms = int(tabla_filtrada.iloc[0]["distancia"])
    else:
        #print("No se encontraron coincidencias.")
        distanciaKms = 0

    #print("Distancia encontrada:", distanciaKms)

    return distanciaKms

# def obtener_distancia(df, origen, destino):
#     if origen != "":
#         fila = df.loc[(df['origen'] == origen) & (df['destino'] == destino), 'distancia']
#         #print(f"distancia: {origen}-{destino}")
#         salida = fila.iloc[0] if not fila.empty else 0
#     else:
#         salida = 0
#     return salida

#@st.cache_resource
def cargar_retriever():
    """
    Carga el retriever híbrido y lo guarda en el estado de la sesión.
    Si ya está cargado, no hace nada.
    """
    #if 'hybrid_retriever' not in st.session_state:
    if st.session_state.hybrid_retriever is None:
        #sparse_retriever = SparseRetriever(documentos)
        #dense_retriever = DenseRetriever(documentos)
        #hybrid_retriever = HybridRetriever(documentos, weight_sparse=0.5, weight_dense=0.5)
        st.session_state.hybrid_retriever = HybridRetriever(documentos, weight_sparse=0.5, weight_dense=0.5)
        print("Retriever híbrido cargado y guardado en el estado de la sesión.")
    else:
        print("El retriever híbrido ya está cargado en el estado de la sesión.")

@st.cache_resource
def descargar_recursos_NLTK():
    print("Descargando recursos de NLTK...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("Recursos NLTK descargados correctamente.")


def evaluar_respuesta(respuesta_generada: str, context: str, query: str):
    """
    Evalúa la respuesta generada por el modelo.
    Compara la respuesta con el contexto proporcionado.
    """
    # Verificar que la respuesta y el contexto no estén vacíos
    if not respuesta_generada or not context:
        st.warning("La respuesta generada o el contexto están vacíos. Por favor, verifica la información.")
        return

    #print(f"Contexto: {query}")
    #print(f"Contexto: {context}")
    #print(f"Respuesta generada: {respuesta_generada}")
    assert query, "Pregunta: está vacía"
    assert context, "Contexto: está vacío"
    assert respuesta_generada, "Respuesta generada: está vacía"

    # Cargar el modelo de evaluación
    model_Eval = SentenceTransformer(ModeloEvaluacion)

    # Generar embeddings
    context_embedding = model_Eval.encode(context, convert_to_numpy=True, normalize_embeddings=True)
    response_embedding = model_Eval.encode(respuesta_generada, convert_to_numpy=True, normalize_embeddings=True)

    # Normalizar los embeddings
    #context_embedding = context_embedding / np.linalg.norm(context_embedding)
    #response_embedding = response_embedding / np.linalg.norm(response_embedding)

    # Calcular similitud (coseno)
    #similarity = np.dot(context_embedding, response_embedding) / (np.linalg.norm(context_embedding) * np.linalg.norm(response_embedding))
    similarity = np.dot(context_embedding, response_embedding)
    print("Similitud entre contexto y respuesta:", similarity)

    # from bert_score import score
    # P, R, F1 = score([respuesta_generada], [context], lang="es")
    # print(f"BERTScore (F1): {F1.mean().item()}")

    # from rouge_score import rouge_scorer
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    # scores = scorer.score(context, respuesta_generada)
    # print(f"ROUGE-L score: {scores['rougeL'].fmeasure}")

    # Si la similitud es alta, la respuesta es relevante
    mensaje = f"Petición del usuario: {query}"
    if similarity > 0.7:  # Umbral de similitud, puedes ajustarlo
        #st.success("La respuesta generada es relevante y está basada en la información proporcionada.")
        st.success(mensaje)
    else:
        #st.warning("La respuesta generada puede no estar completamente alineada con la información proporcionada. Por favor, verifica la respuesta.")
        st.warning(mensaje)

    print("Precisión de la respuesta: {similarity:.2%}")

# Función para enviar la consulta
def consultar_gemini(context, query):
    headers = {"Content-Type": "application/json"}

    # Recupera los documentos relevantes
    #retrieved_docs_text = [corpus[idx] for idx in I[0]]
    # También recuperamos las filas completas del DataFrame para más detalles si son necesarios
    # retrieved_docs_df = df.iloc[I[0]]
    #context = "\n\n".join(corpus)
    #context = corpus

    # Crear un prompt más detallado para guiar a Gemini
    prompt_template = f"""
    Eres un asistente experto en ciclos formativos en Extremadura.
    Aquí tienes información relevante sobre ciclos formativos en Extremadura:

    ---
    {context}
    ---

    Basándote ÚNICAMENTE en la información proporcionada anteriormente y en tu conocimiento general, responde a la siguiente pregunta de forma concisa y útil.
    Contexta solo si tienes evidencia directa en la información proporcionada.
    Si no tienes información suficiente, indica que no puedes responder.
    Tu respuesta debe ser clara, estructurada y contener solo la información relevante.
    No inventes información, no especules ni hagas suposiciones.
    Si la información proporcionada no es suficiente para responder a la pregunta, indica que no puedes responder.
    Si la pregunta es sobre ciclos formativos, institutos, grados, turnos, municipios o provincias, responde con la información más precisa y detallada posible.
    Solo incluye en tu respuesta los ciclos formativos que coincidan directamente con la consulta del usuario y que estén presentes en la información proporcionada.
    Si la información proporcionada no es suficiente para responder a la pregunta, indícalo.
    Muestra la información de forma clara y estructurada por instituto y ciclo formativo.
    Mostrar los detalles como instituto, nombre del ciclo, grado, turno, municipio (provincia), si son relevantes y no duplican la información, en formato tabla.
    Ordenar la salida por instituto y nombre del ciclo.
    Un ciclo es nuevo si en el campo "Tipo" está marcado como "Nuevo".

    Pregunta: {query}

    Respuesta:
    """

    # payload = {
    #     "prompt": prompt_template,  # Texto que quieres procesar
    #     "temperature": 0.7,  # Configurar creatividad de la respuesta
    #     "max_tokens": 500  # Limitar longitud de la respuesta
    # }

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_template}]}]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    result = response.json()

    if response.status_code == 200:
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Lo siento, no pude obtener una respuesta de la IA. La estructura de la respuesta fue inesperada."
    else:
        return f"Error {response.status_code}: {response.text}"

# # Prueba con un corpus de ejemplo
# corpus = "¿Cuáles son los ciclos formativos de Comercio disponibles en Extremadura?"
# respuesta = consultar_gemini(corpus, user_query)

# # Mostrar respuesta
# print(json.dumps(respuesta, indent=2, ensure_ascii=False))

# -------------------------------------------------------------------
# --- Inicio de aplicación ---
# Preparación de los datos
descargar_logo(FileLogo) # Descargar logo del chatbot, si no existe
descargar_pdf(FilePDF) # Descargar el PDF, si no existe
extraer_informacion_pdf(FilePDF) # Extraer información del PDF
descargar_distancias(FileDistancias)  # Descargar el fichero de distancias, si no existe

# -------------------------------------------------------------------
# --- Configuración inicial de la aplicación Streamlit ---
st.set_page_config(page_title="Chatbot de Ciclos Formativos", layout="centered")

# Preparar el entorno de la aplicación
mostrar_logo_titulo()  # Mostrar el logo del chatbot
inicializar_entorno()   # Inicializar el entorno de la aplicación

# Inicializar df leyendo el Excel si existe, si no, se creará más adelante
if os.path.exists(FileExcel):
    df = pd.read_excel(FileExcel)
    st.session_state.excel_data = df
else:
    df = pd.DataFrame()

# Mostrar las primeras filas del DataFrame para verificar que se ha cargado correctamente
#st.write(df.head())
#st.dataframe(df.head())  # Alternativa para mostrar el DataFrame de forma interactiva
#st.write("Datos cargados desde el archivo Excel existente.")
# -------------------------------------------------------------------

# Distancias entre localidades
if os.path.exists(FileDistancias):
    distancias = pd.read_excel(FileDistancias)
    #distancias.columns = distancias.columns.str.strip()  # Elimina espacios en nombres de columnas
    distancias.rename(columns={"Distancia (Km)": "distancia"}, inplace=True)
    # Seleccionar las columnas deseadas
    distancias = distancias[["origen", "destino", "distancia"]]
    st.session_state.excel_kms = distancias
else:
    distancias = pd.DataFrame()

# FileDistanciasMini = "DistanciasMini.xlsx"
# distancias.to_excel(FileDistanciasMini, index=False)
# print("Fichero Excel creado : " + FileDistanciasMini)

#print(distancias.head())
xDistancias = False
Localidad = ""
print("Distancias: ", xDistancias, Localidad)
#st.success(f"Distancias: {xDistancias}, {Loc_origen}")

print("Inicializando sistema RAG..")
descargar_recursos_NLTK()  # Descargar recursos de NLTK necesarios para el preprocesamiento

# -------------------------------------------------------------------

# Inicializamos el retriever hybrido
documentos = df[['Familia', 'Ciclo', 'Grado', 'Instituto', 'Municipio', 'Provincia', 'Turno', 'Tipo']].apply(lambda x: f"{x['Familia']} - {x['Ciclo']} - {x['Grado']} - {x['Instituto']} - {x['Municipio']} - {x['Provincia']} - {x['Turno']} - {x['Tipo']}", axis=1).tolist()
cargar_retriever()  # Cargar el retriever híbrido y guardarlo en el estado de la sesión

# documentos = df['descripcion'].tolist()
# documentos = df['Familia Profesional'].tolist()
#sparse_retriever = SparseRetriever(documentos)
#dense_retriever = DenseRetriever(documentos)
#hybrid_retriever = HybridRetriever(documentos, weight_sparse=0.5, weight_dense=0.5)
# -------------------------------------------------------------------

# cargar_modelo()  # Cargar el modelo de embeddings
# cargar_datos_indice_FAISS(df)  # Cargar los datos del DataFrame y crear el índice FAISS
if st.session_state.hybrid_retriever is not None:
    st.sidebar.success("¡Chatbot iniciado correctamente!")  # Mensaje de éxito en la barra lateral

# Mostrar historial de chat
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# --- Configuración de la BARRA LATERAL: sidebar ---
#st.image(image, caption='Chatbot de Ciclos Formativos', use_column_width=True)

# Mostrar información del chatbot
st.sidebar.header(TituloAPP)
st.sidebar.markdown("""
    Puedes realizar preguntas sobre:
    - Ciclos formativos disponibles
    - Institutos/centros educativos
    - Familias profesionales
    - Grados/niveles de formación
    - Turnos: diurno o vespertino, nuevos ciclos
    - Centros más cercanos para estudiar.
""")

ModoE = "Escribir"
modoE = st.sidebar.radio("Elige el modo de entrada:", ("Escribir", "Hablar"))
Localidad = st.sidebar.text_input("Localidad de origen:", "")

# Preguntar al usuario cómo quiere interactuar con el chatbot
if modoE == "Hablar":
    st.write("La grabación se inicia al pulsar el botón del micrófono, y finaliza al volver a pulsar el botón.")
    colVoz1, colVoz2 = st.columns([1,10])
    with colVoz1:
        user_query = grabar_audio()  # Grabar audio y convertirlo a texto
    with colVoz2:
        if user_query:
            st.write("🗣️ ", user_query) # Mostrar la transcripción del audio
else:
    user_query = st.chat_input("Haz tu pregunta sobre los ciclos formativos...")

if user_query:
    xDistancias = (Localidad != "")
    if xDistancias:
        st.success(f"Petición: {user_query} (ordenado por distacias desde {Localidad})")
    else:
        st.success(f"Petición: {user_query}")
    lanzar_consulta(user_query)
else:
     print("No hay petición del usuario.")
# -------------------------------------------------------------------

# Mostrar información del archivo PDF y Excel
# show_datos = st.sidebar.checkbox("¿Mostrar datos utilizados?")
# if show_datos:
#     #st.sidebar.subheader("Información utilizada")
#     st.sidebar.write(f"- Fuente: `{FilePDF}`")
#     #st.sidebar.write(f"- `{FileExcel}`")

#     if st.session_state.excel_data is not None:
#         st.sidebar.write(f"- Nº Ciclos Formativos: {len(st.session_state.excel_data):,.0f}".replace(",", "."))

    #if st.session_state.model is not None:
    #    st.sidebar.write(f"- Modelo: `{ModeloEmbeddings}`")

# new_pdf = st.sidebar.checkbox("¿Cargar nuevo PDF de datos?")
# if new_pdf:
#     # Cargar PDF
#     pdf_obj = st.sidebar.file_uploader("Carga el documento PDF fuente", type="pdf")
#     # Si se carga un PDF, procesarlo
#     if pdf_obj is not None:
#         # Guardar el PDF en un archivo temporal
#         with open(FilePDF, "wb") as f:
#             f.write(pdf_obj.getbuffer())
#         # Extraer información del PDF y crear el DataFrame
#         df = extraer_informacion_pdf(FilePDF)
#         st.session_state.excel_data = df
#         st.session_state.faiss_index, st.session_state.corpus = create_faiss_index(df, st.session_state.model)
#         st.success("¡Datos cargados, embeddings e índice FAISS creados correctamente! Ahora puedes hacer preguntas.")
#         # Limpiar historial de chat al cargar un nuevo archivo
#         st.session_state.chat_history = []

# Mostrar el DataFrame cargado desde el PDF
# if st.session_state.excel_data is not None:
#     st.subheader("Datos Cargados desde el PDF")
#     st.dataframe(st.session_state.excel_data.head())  # Mostrar las primeras filas del DataFrame
# else:
#     st.info("Por favor, sube un archivo PDF para empezar a interactuar con el chatbot.")

#show_historial = st.sidebar.checkbox("¿Mostrar el Historial del Chat?")
#if show_historial:
#    st.sidebar.subheader("Historial de Chat")
#    if st.session_state.chat_history:
#        st.sidebar.write(f"Total de mensajes en el historial: {len(st.session_state.chat_history)}")
#        if len(st.session_state.chat_history) > 0:
#            st.sidebar.write("Último mensaje:")
#            last_message = st.session_state.chat_history[-1]
#            st.sidebar.write(f"{last_message['role']}: {last_message['content']}")
#    else:
#        st.sidebar.write("No hay mensajes en el historial de chat.")

# Opcional: Botón para limpiar el historial de chat
# if st.sidebar.button("Vaciar Chat"):
#     st.session_state.chat_history = []  # Reiniciar el historial de chat
#     st.session_state.messages = []  # Reiniciar los mensajes
#     st.empty()  # Limpia la pantalla de chat
#     st.success("Chat vaciado. Puedes empezar de nuevo.")

# if st.sidebar.button("Reiniciar Chat"):
#     st.session_state.clear()  # Borra todas las variables de sesión
#     st.empty()  # Limpia la pantalla de chat
#     st.session_state.chat_history = []  # Reiniciar el historial de chat
#     st.rerun()
#     st.success("Chat reiniciado. Puedes empezar de nuevo.")

# Footer
st.sidebar.markdown("""
    **Desarrollado por:**
    - Antonio Jesús Abasolo Sierra
    - José David Acedo Honrado
""")

# Leer el archivo en modo binario
with open(FileExcel, "rb") as f:
    excel_bytes = f.read()

# Botón de descarga en Streamlit
st.sidebar.download_button(
    label="Descargar archivo Excel 📂",
    data=excel_bytes,
    file_name=FileExcel,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# with open(FileHybrida, "rb") as f:
#     excel_bytes = f.read()

# st.sidebar.download_button(
#     label="Descargar archivo Hybrida.xlsx 📂",
#     data=excel_bytes,
#     file_name=FileHybrida,
#     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# )
# with open(FileDistanciasMini, "rb") as f:
#     excel_bytes = f.read()

# st.sidebar.download_button(
#     label="Descargar archivo DistanciasMini.xlsx 📂",
#     data=excel_bytes,
#     file_name=FileDistanciasMini,
#     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# )

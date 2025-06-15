# -------------------------------------------------------------------
#pip install streamlit pandas faiss-cpu sentence-transformers requests
#pip install openai langchain PyPDF2 langdetect langchain-community sckit-learn
#pip install spacy scikit-learn openpyxl pymupdf gtts
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Importar las librerías necesarias
# -------------------------------------------------------------------
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import os
import requests
import time

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

# -------------------------------------------------------------------
# --- Definición de rutas y ficheros de datos ---
url = "https://raw.githubusercontent.com/antoniojasweb/chatbot/main/pdf/"
FilePDF = "25_26_OFERTA_por_Familias.pdf"
FileExcel = "oferta_formativa_completa.xlsx"
FileLogo = "Logo.png"

# --- Modelo de embeddings ---
# Otro modelo de Sentence Transformers, para Ingés: 'all-MiniLM-L6-v2'
ModeloEmbeddings = 'paraphrase-multilingual-MiniLM-L12-v2'
ModeloEvaluacion = 'all-mpnet-base-v2'  # Modelo para evaluación de respuestas
# Otros posibles modelos: all-MiniLM-L6-v2, sentence-t5-base, e5-large-v2, etc

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
    bilingue = "No"
    nuevo = "No"

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
                        if 'Vespertino' in text:
                            turno = 'Vespertino'
                        elif 'Diurno' in text:
                            turno = 'Diurno'
                        elif text in ['Bilingüe', 'Bilingue']:
                            bilingue = 'Sí'
                        elif 'Nuevo' in text:
                            nuevo = 'Sí'

                        # Añadir fila: Sólo si el curso es de 1ºC
                        if "1ºC" in curso:
                            data.append({
                                "Familia Profesional": familia_actual.title().strip(),
                                "Código Ciclo": codigo_ciclo.strip(),
                                "Nombre Ciclo": nombre_ciclo.title().strip(),
                                "Grado": grado_actual.strip(),
                                "Instituto": instituto.strip(),
                                "Municipio": municipio.title().strip(),
                                "Provincia": provincia_actual.title().strip(),
                                #"Curso": curso.strip(),
                                "Turno": turno,
                                "Bilingüe": bilingue,
                                "Nuevo": nuevo
                            })

    if data:
        # Convertir a DataFrame
        df = pd.DataFrame(data)

        # ordenar por la columna Municipio y Código Ciclom, en orden ascendente
        df.sort_values(['Familia Profesional','Código Ciclo'], ascending=[True, True], inplace=True)

        # Exportar a Excel
        df.to_excel(FileExcel, index=False)
        print("Fichero Excel creado : " + FileExcel)
    else:
        print("No se encontraron datos válidos en el PDF.")
        df = pd.DataFrame(columns=[
            "Familia Profesional", "Código Ciclo", "Nombre Ciclo", "Grado",
            "Instituto", "Municipio", "Provincia", "Turno", "Bilingüe", "Nuevo"
        ])

    # Si quieres mostrar el DataFrame en Streamlit, puedes descomentar la siguiente línea
    #st.write(df)

    # Mostrar primeras filas
    #print(df.head())
    #df.head()

    return df
#--------------------------------------------------------------------

# --- Cargar el modelo de embeddings y crear el índice FAISS ---
@st.cache_resource
def load_embedding_model():
    """
    Carga el modelo de embeddings pre-entrenado.
    """
    #st.write("Cargando modelo de embeddings (esto puede tardar unos segundos)...")
    model = SentenceTransformer(ModeloEmbeddings)
    print("Modelo de embeddings cargado: " + ModeloEmbeddings)
    return model

def create_faiss_index(df: pd.DataFrame, model: SentenceTransformer):
    """
    Crea un índice FAISS a partir de los datos del DataFrame.
    """
    #st.write("Creando índice FAISS...")
    # Concatenar las columnas relevantes en una sola cadena de texto para el embedding
    df['combined_text'] = df.apply(
        lambda row: f"Nombre Ciclo: {row.get('Nombre Ciclo', '')}. Grado: {row.get('Grado', '')}. Familia Profesional: {row.get('Familia Profesional', '')}. Instituto: {row.get('Instituto', '')}. Municipio: {row.get('Municipio', '')}. Provincia: {row.get('Provincia', '')}. Turno: {row.get('Turno', '')}",
        axis=1
    )

    # Rellenar cualquier NaN con cadena vacía para evitar errores de embedding
    df['combined_text'] = df['combined_text'].fillna('')

    corpus = df['combined_text'].tolist()
    #embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    embeddings = model.encode(corpus, show_progress_bar=True)
    # Normalizar los embeddings, para mejorar la búsqueda de similitud: L2 normalization
    embeddings_normalized = normalize(embeddings, norm='l2')

    # Escalar los embeddings para mejorar la precisión de FAISS: Opcional, pero recomendado
    # Si los embeddings ya están normalizados, este paso es redundante
    # Aunque FAISS no requiere escalado, puede ayudar en algunos casos
    # Normalizar los embeddings para que tengan media 0 y desviación estándar 1
    # Esto es opcional, pero puede mejorar la precisión de búsqueda
    #scaler = StandardScaler()
    #embeddings_scaled = scaler.fit_transform(embeddings_normalized)

    # Crear un índice FAISS Flat (Simple)
    dimension = embeddings_normalized.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings_normalized).astype('float32')) # FAISS requiere float32

    print("Índice FAISS creado: embeddings normalizado con L2.")
    return index, corpus # Devolvemos también el corpus para poder mapear los resultados

def get_gemini_response(prompt: str):
    """
    Hace una llamada a la API de Gemini para obtener una respuesta.
    """
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Lanza una excepción para errores HTTP
        result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Lo siento, no pude obtener una respuesta de la IA. La estructura de la respuesta fue inesperada."
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API de Gemini: {e}. Por favor, verifica tu conexión y clave API.")
        return "Lo siento, hubo un problema al conectar con el servicio de IA."
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
        return "Lo siento, ocurrió un error inesperado al procesar tu solicitud."

def ask_rag_model(query: str, index, corpus: list, model: SentenceTransformer, df: pd.DataFrame, top_k: int = 10):
    """
    Realiza la consulta RAG:
    1. Embed de la consulta.
    2. Busca documentos relevantes en el índice FAISS.
    3. Construye un prompt contextualizado para el LLM.
    4. Llama al LLM para generar la respuesta.
    """

    if top_k == 0:
        top_k = len(corpus)

    retrieved_docs_text = []
    retrieved_docs_df= pd.DataFrame()

    query_embedding = model.encode([query]).astype('float32')

    # Realiza la búsqueda en FAISS
    D, I = index.search(query_embedding, top_k) # D es la distancia, I son los índices

    # Recupera los documentos relevantes
    retrieved_docs_text = [corpus[idx] for idx in I[0]]
    # También recuperamos las filas completas del DataFrame para más detalles si son necesarios
    retrieved_docs_df = df.iloc[I[0]]

    context = "\n\n".join(retrieved_docs_text)

    # Crear un prompt más detallado para guiar a Gemini
    prompt_template = f"""
    Eres un asistente experto en ciclos formativos en Extremadura.
    Aquí tienes información relevante sobre ciclos formativos en Extremadura:

    ---
    {context}
    ---

    Basándote ÚNICAMENTE en la información proporcionada anteriormente y en tu conocimiento general, responde a la siguiente pregunta de forma concisa y útil. Si la información proporcionada no es suficiente para responder a la pregunta, indícalo. Muestra la información de forma clara y estructurada por instituto y cliclo formativo, incluyendo detalles como el nombre del ciclo, grado, instituto, municipio, provincia y familia profesional si es relevante.

    Pregunta: {query}

    Respuesta:
    """

    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": f"Buscando información relacionada con '{query}'..."})

    # Muestra los documentos recuperados para depuración o información al usuario
    #with st.expander("Ver información recuperada: " + str(top_k) + " opciones más relevantes"):
    #    st.write(retrieved_docs_df[['Nombre Ciclo', 'Grado', 'Instituto', 'Municipio', 'Provincia', 'Familia Profesional']])
        #print(retrieved_docs_df[['Nombre Ciclo', 'Grado', 'Instituto', 'Municipio', 'Provincia', 'Familia Profesional']])
    print("Documentos recuperados:")
    print(retrieved_docs_df)

    print("Los datos encontrados a tu pregunta son:")
    print(context)

    evaluar_respuesta(context, query)

    return get_gemini_response(prompt_template)

def evaluar_respuesta(respuesta_generada: str, context: str):
    """
    Evalúa la respuesta generada por el modelo.
    Compara la respuesta con el contexto proporcionado.
    """
    # Verificar que la respuesta y el contexto no estén vacíos
    if not respuesta_generada or not context:
        st.warning("La respuesta generada o el contexto están vacíos. Por favor, verifica la información.")
        return

    print(f"Respuesta generada: {respuesta_generada}")
    print(f"Contexto: {context}")
    assert respuesta_generada, "Respuesta generada: está vacía"
    assert context, "Contexto: está vacío"

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
    if similarity > 0.6:  # Umbral de similitud, puedes ajustarlo
        #st.success("La respuesta generada es relevante y está basada en la información proporcionada.")
        st.success(" ")
    else:
        #st.warning("La respuesta generada puede no estar completamente alineada con la información proporcionada. Por favor, verifica la respuesta.")
        st.warning(" ")
# -------------------------------------------------------------------

# --- Función para convertir texto a audio y obtenerlo en base64 ---
def text_to_audio_base64(text, lang='es'):
    """
    Convierte texto a audio usando gTTS y devuelve el audio codificado en base64.
    """
    try:
        text = text.replace('*', ' ')
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        # Codificar el buffer de audio en base64
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        # Limpiar el buffer
        audio_buffer.close()
        # Devolver el audio en formato base64
        print("Audio generado y codificado en base64.")
        return audio_base64
    except Exception as e:
        print(f"Error al generar audio: {e}")
        return None

def grabar_audio2():
    #st.write("Puedes grabar tu pregunta usando el micrófono.")
    st.write("La grabación se inicia al pulsar el botón del micrófono, y finaliza al volver a pulsar el botón.")
    col1, col2 = st.columns([1,12])
    with col1:
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
                        with col2:
                            # Mostrar la transcripción del audio
                            st.write("🗣️ Transcripción:", text)
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


def grabar_audio():
    """
    Graba audio desde el micrófono y lo convierte a texto.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Grabando...")
        audio = recognizer.listen(source)
        st.write("Grabación finalizada.")
        try:
            text = recognizer.recognize_google(audio, language='es-ES')
            with col2:
                # Mostrar la transcripción del audio
                st.write("🗣️ ==> ", text)
            #st.write(f"Texto reconocido: {text}")
            return text
        except sr.UnknownValueError:
            st.error("No se pudo reconocer el audio.")
            return None
        except sr.RequestError as e:
            st.error(f"Error al conectar con el servicio de reconocimiento de voz: {e}")
            return None

def Lanzar_pregunta(user_query):
    """
    Lanza la pregunta al modelo RAG y muestra la respuesta.
    """
    if user_query:
        # Mostrar mensaje de búsqueda
        st.write(f"Buscando información relacionada con: {user_query}")
        # Llamar al modelo RAG para obtener la respuesta
        respuesta = ask_rag_model(user_query, st.session_state.faiss_index, st.session_state.corpus, st.session_state.model, st.session_state.excel_data)
        # Mostrar la respuesta generada
        with st.chat_message("assistant"):
            st.write(respuesta)
            # Convertir la respuesta a audio y mostrarla
            audio_base64 = text_to_audio_base64(respuesta)
            if audio_base64:
                audio_html = f'<audio controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg"></audio>'
                st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.error("No se pudo generar el audio de la respuesta.")
        # Añadir al historial de chat
        st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
    else:
        st.warning("Por favor, escribe o graba una pregunta antes de enviar.")

def Lanzar_consulta(user_query):
    """
    Lanza la consulta al modelo RAG y muestra la respuesta.
    """
    # Entrada de usuario
    if st.session_state.excel_data is not None and st.session_state.faiss_index is not None:
        # Entrada de voz del usuario (opcional)
        # modo = st.radio("Elige el modo de entrada:", ("Escribir", "Hablar"))
        # if modo == "Hablar":
        #     user_query = grabar_audio()  # Grabar audio y convertirlo a texto
        # else:
        #     user_query = st.chat_input("Haz tu pregunta sobre los ciclos formativos...")

        # Reemplazar términos específicos en la consulta del usuario para mejorar la búsqueda
        #consulta_modificada = consulta_usuario.replace("Ciudad", "Municipio")
        #consulta_modificada = consulta_usuario.replace("Vespertino", "Tarde")
        equivalencias = {"Ciudad": "Municipio",
                        "Localidad": "Municipio",
                        "Vespertino": "Tarde",
                        "Bilingüe": "Bilingue",
                        "Bilingue": "Bilingüe",
                        "Nuevo": "Nuevo Ciclo",
                        "Ciclo Formativo": "Ciclo",
                        "Instituto": "Centro Educativo"}

        if user_query:
            # Reemplazar términos en la consulta del usuario
            user_query = user_query.strip()  # Limpiar espacios al inicio y final
            user_query = user_query.replace("?", "")  # Eliminar signos de interrogación
            user_query = user_query.replace("¿", "")  # Eliminar signos de interrogación al inicio
            user_query = user_query.replace(".", "")  # Eliminar puntos al final
            user_query = user_query.replace(",", "")  # Eliminar comas al final
            user_query = user_query.replace(":", "")  # Eliminar dos puntos al final
            user_query = user_query.replace(";", "")  # Eliminar punto y coma al final
            user_query = user_query.replace("  ", " ")  # Eliminar dobles espacios

            # Reemplazar equivalencias en la consulta del usuario
            for clave, valor in equivalencias.items():
                user_query = user_query.replace(clave, valor)

            with st.spinner("Pensando...", show_time=True):
                response = ask_rag_model(
                    user_query,
                    st.session_state.faiss_index,
                    st.session_state.corpus,
                    st.session_state.model,
                    st.session_state.excel_data
                )

            with st.chat_message("assistant"):
                st.write(response)
                # Convertir respuesta del bot a audio base64
                audio_b64 = text_to_audio_base64(response, lang='es')
                if audio_b64:
                    st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")

            # Añadir la respuesta al historial de chat
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# -------------------------------------------------------------------
# --- Comprobación de existencia de archivos y carga de datos ---
# Comprobar si el archivo PDF existe, si no, descargarlo
if not os.path.exists(FilePDF):
    descargar_pdf(FilePDF)

# Extraer información del PDF y crear el DataFrame
df = extraer_informacion_pdf(FilePDF)

# Descargar el logo del chatbot si no existe
if not os.path.exists(FileLogo):
    descargar_logo(FileLogo)
image = Image.open(FileLogo)

# Mostrar las primeras filas del DataFrame para verificar que se ha cargado correctamente
#st.write(df.head())
#st.dataframe(df.head())  # Alternativa para mostrar el DataFrame de forma interactiva
#st.write("Datos cargados desde el archivo Excel existente.")
#df.head()
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# --- Configuración de la aplicación Streamlit ---
st.set_page_config(page_title="Chatbot de Ciclos Formativos", layout="centered")

# Mostrar el logo del chatbot
#st.image(image, caption='Chatbot-FP', width=200)
#st.title("📚 Chatbot de Ciclos Formativos")

col1, col2 = st.columns([1,2])
with col1:
    st.image(image, width=150)
with col2:
    st.title("Ciclos Formativos en Extremadura")

# Inicializar el estado de la sesión si no existe
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "model" not in st.session_state:
    st.session_state.model = None
if "corpus" not in st.session_state:
    st.session_state.corpus = None

# Cargar el modelo de embeddings solo una vez
if st.session_state.model is None:
    # Mostrar mensaje de preparación del entorno
    st.write("Preparando el entorno (esto puede tardar unos segundos)...")
    st.session_state.model = load_embedding_model()

if st.session_state.excel_data is None:
    try:
        # Asegurarse de que las columnas esperadas existan o manejar su ausencia
        required_cols = ['Familia Profesional', 'Grado', 'Código Ciclo', 'Nombre Ciclo', 'Instituto', 'Municipio', 'Provincia', 'Turno', 'Bilingüe', 'Nuevo']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Columnas faltantes en el Excel: {', '.join(missing_cols)}. El chatbot podría no funcionar correctamente.")
            # Intentar crear las columnas faltantes con valores vacíos para que el script no falle
            for col in missing_cols:
                df[col] = ''

        st.session_state.excel_data = df
        st.session_state.faiss_index, st.session_state.corpus = create_faiss_index(st.session_state.excel_data, st.session_state.model)
        st.success("¡Chatbot iniciado correctamente! Ahora puedes hacer preguntas.")

        # Limpiar historial de chat al cargar un nuevo archivo
        st.session_state.chat_history = []
    except Exception as e:
        st.error(f"Error al leer el archivo de datos o crear el índice: {e}")
        st.session_state.excel_data = None
        st.session_state.faiss_index = None
        st.session_state.corpus = None

# Mostrar historial de chat
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# Preguntar al usuario cómo quiere interactuar con el chatbot
modo = st.radio("Elige el modo de entrada:", ("Escribir", "Hablar"))
if modo == "Hablar":
    user_query = grabar_audio2()  # Grabar audio y convertirlo a texto
else:
    user_query = st.chat_input("Haz tu pregunta sobre los ciclos formativos...")
Lanzar_consulta(user_query)

# -------------------------------------------------------------------

# --- Configuración de la BARRA LATERAL: sidebar ---
#st.image(image, caption='Chatbot de Ciclos Formativos', use_column_width=True)

# Mostrar información del chatbot
st.sidebar.header("Chatbot de Ciclos Formativos en Extremadura")
st.sidebar.markdown("""
    Este chatbot te permite hacer preguntas sobre los ciclos formativos en Extremadura basándose en datos extraídos del PDF indicado. \n
    Puedes preguntar sobre:
    - Ciclos formativos disponibles
    - Institutos y centros educativos
    - Familias profesionales
    - Grados y niveles de formación
    - Información sobre turnos y modalidades (diurno, vespertino, bilingüe, etc.)
    - Y mucho más relacionado con la oferta formativa en Extremadura.
    \n\n
""")

# Mostrar información del archivo PDF y Excel
show_datos = st.sidebar.checkbox("¿Mostrar datos utilizados?")
if show_datos:
    #st.sidebar.subheader("Información utilizada")
    st.sidebar.write(f"- Fuente: `{FilePDF}`")
    #st.sidebar.write(f"- `{FileExcel}`")

    if st.session_state.excel_data is not None:
        st.sidebar.write(f"- Nº Ciclos Formativos: {len(st.session_state.excel_data):,.0f}".replace(",", "."))

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
if st.sidebar.button("Vaciar Chat"):
    st.session_state.chat_history = []  # Reiniciar el historial de chat
    st.session_state.messages = []  # Reiniciar los mensajes
    st.empty()  # Limpia la pantalla de chat
    st.success("Chat vaciado. Puedes empezar de nuevo.")

if st.sidebar.button("Reiniciar Chat"):
    st.session_state.clear()  # Borra todas las variables de sesión
    st.empty()  # Limpia la pantalla de chat
    st.session_state.chat_history = []  # Reiniciar el historial de chat
    st.rerun()
    st.success("Chat reiniciado. Puedes empezar de nuevo.")

# Footer
st.sidebar.markdown("""
    ---
    **Desarrollado por:**
    - Antonio J. Abasolo Sierra
    - José David Honrado García
""")

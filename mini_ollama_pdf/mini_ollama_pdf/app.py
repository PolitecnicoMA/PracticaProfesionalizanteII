from flask import Flask, render_template, request, jsonify
from ollama import chat, embed
from PyPDF2 import PdfReader

import os
import math
import time

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"

# memoria simple del documento
document_chunks = []
document_vectors = []

LLM_MODEL = "qwen2.5:0.5b"
EMBED_MODEL = "embeddinggemma"


# ------------------------------------------------
# EXTRAER TEXTO DEL PDF
# ------------------------------------------------

def extract_text(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


# ------------------------------------------------
# DIVIDIR TEXTO
# ------------------------------------------------

def split_chunks(text, size=2000):

    text = text.replace("\n", " ")

    chunks = []

    for i in range(0, len(text), size):

        chunk = text[i:i + size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks[:10]


# ------------------------------------------------
# COSINE SIMILARITY
# ------------------------------------------------

def cosine_similarity(v1, v2):

    dot = sum(a*b for a,b in zip(v1,v2))

    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))

    if n1 == 0 or n2 == 0:
        return 0

    return dot/(n1*n2)


# ------------------------------------------------
# CREAR EMBEDDINGS
# ------------------------------------------------

def create_embeddings(chunks):

    vectors = []

    for chunk in chunks:

        e = embed(
            model = EMBED_MODEL,
            input = chunk
        )

        vectors.append(e["embeddings"][0])

    return vectors


# ------------------------------------------------
# BUSQUEDA SEMANTICA
# ------------------------------------------------

def search(question):

    q_vec = embed(
        model = EMBED_MODEL,
        input = question
    )["embeddings"][0]

    scores = []

    for chunk,vec in zip(document_chunks, document_vectors):

        s = cosine_similarity(q_vec,vec)

        scores.append((s,chunk))

    scores.sort(reverse=True,key=lambda x:x[0])

    return scores[:3]


# ------------------------------------------------
# HOME
# ------------------------------------------------

@app.route("/")
def index():

    return render_template("index.html")


# ------------------------------------------------
# SUBIR PDF
# ------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():

    global document_chunks, document_vectors

    pdf = request.files["pdf_file"]

    os.makedirs("uploads", exist_ok=True)

    path = os.path.join("uploads", pdf.filename)

    pdf.save(path)

    text = extract_text(path)

    document_chunks = split_chunks(text)

    document_vectors = create_embeddings(document_chunks)

    return jsonify({
        "chunks": len(document_chunks)
    })


# ------------------------------------------------
# CHAT
# ------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat_api():

    data = request.json
    question = data["message"]

    steps = []

    # Paso 1
    steps.append("Pregunta recibida")

    # Paso 2: embedding de la pregunta
    q_vec = embed(
        model = EMBED_MODEL,
        input = question
    )["embeddings"][0]

    steps.append("Embedding de la pregunta generado")

    # Paso 3: búsqueda semántica
    scores = []

    for chunk,vec in zip(document_chunks, document_vectors):

        s = cosine_similarity(q_vec,vec)

        scores.append((s,chunk))

    scores.sort(reverse=True,key=lambda x:x[0])

    top_chunks = [chunk for score,chunk in scores[:3]]

    steps.append("Búsqueda semántica en el documento")

    context = "\n\n".join(top_chunks)

    steps.append("Contexto construido para el modelo")

    prompt = f"""
Usa el siguiente contexto del documento para responder.

Contexto:
{context}

Pregunta:
{question}
"""

    start = time.time()

    response = chat(
        model = LLM_MODEL,
        messages=[
            {"role":"user","content":prompt}
        ],
        options={
            "temperature":0.2,
            "num_predict":120
        }
    )

    elapsed = round(time.time() - start, 2)

    steps.append("Respuesta generada por el modelo")

    answer = response["message"]["content"].strip()

    return jsonify({
        "answer": answer,
        "time": elapsed,
        "steps": steps
    })


if __name__ == "__main__":

    app.run(debug=True)
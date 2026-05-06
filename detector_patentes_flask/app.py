from flask import Flask, render_template, Response, jsonify
import cv2
import re
from ultralytics import YOLO
import easyocr

app = Flask(__name__)

model = YOLO("models/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

video_path = "static/videos/autos.mp4"
cap = cv2.VideoCapture(video_path)

# memoria
patentes_detectadas = set()
patentes_mostrar = []

# estado persistente visual
ultima_caja = None
ultimo_texto = ""

# confirmación mínima
contador_lecturas = {}

frame_count = 0
PROCESS_EVERY = 5
SCALE_DET = 1.8
CONF_YOLO = 0.25
MIN_CONFIRMACIONES = 2


def limpiar(texto):
    texto = texto.upper()
    texto = re.sub(r"[^A-Z0-9]", "", texto)
    return texto


def detectar(frame):
    global ultima_caja, ultimo_texto
    global patentes_detectadas, patentes_mostrar, contador_lecturas

    frame_big = cv2.resize(frame, None, fx=SCALE_DET, fy=SCALE_DET)

    results = model.predict(frame_big, conf=CONF_YOLO, verbose=False)

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            # volver a escala original
            x1 = int(x1 / SCALE_DET)
            y1 = int(y1 / SCALE_DET)
            x2 = int(x2 / SCALE_DET)
            y2 = int(y2 / SCALE_DET)

            # validar caja
            if x2 <= x1 or y2 <= y1:
                continue

            patente = frame[y1:y2, x1:x2]

            if patente.size == 0:
                continue

            gray = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            gray = cv2.equalizeHist(gray)

            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

            resultado = reader.readtext(
                thresh,
                detail=0,
                decoder="beamsearch",
                beamWidth=5,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            if resultado:
                texto = limpiar(resultado[0])

                if 5 <= len(texto) <= 8:
                    ultima_caja = (x1, y1, x2, y2)
                    ultimo_texto = texto

                    contador_lecturas[texto] = contador_lecturas.get(texto, 0) + 1

                    if (
                        contador_lecturas[texto] >= MIN_CONFIRMACIONES
                        and texto not in patentes_detectadas
                    ):
                        patentes_detectadas.add(texto)
                        patentes_mostrar.append(texto)
                        print("Nueva patente confirmada:", texto)

            break  # solo la mejor detección

    # limpiar contador para que no crezca infinito
    if len(contador_lecturas) > 50:
        contador_lecturas = {}


def dibujar(frame):
    if ultima_caja:
        x1, y1, x2, y2 = ultima_caja

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if ultimo_texto:
            cv2.putText(
                frame,
                ultimo_texto,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    return frame


def generar_frames():
    global cap, frame_count

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            continue

        frame_count += 1

        # detecta cada N frames
        if frame_count % PROCESS_EVERY == 0:
            detectar(frame)

        # siempre dibuja
        frame = dibujar(frame)

        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(
        generar_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/patentes')
def get_patentes():
    return jsonify(patentes_mostrar[-10:])


if __name__ == "__main__":
    app.run(debug=True)
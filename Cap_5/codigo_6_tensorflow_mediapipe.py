# Este programa representa la fase final del proyecto: la inferencia en tiempo real.
# Su objetivo es capturar el video desde la camara de la Nvidia Jetson, detectar si hay
# una mano en la imagen, extraer la posicion de sus articulaciones y pasarle esos datos
# a la red neuronal que entrenamos previamente para que adivine el numero que estamos mostrando.

# El flujo de datos que ocurre en cada fotograma es el siguiente:
# 1. OpenCV lee la imagen cruda desde la tuberia optimizada de GStreamer.
# 2. MediaPipe analiza el fotograma y localiza los 21 puntos clave (landmarks) de la mano.
# 3. Las coordenadas x, y, z de estos puntos se extraen y se aplanan en una lista de 63 valores.
# 4. El escalador (StandardScaler) guardado transforma estos valores para que tengan la misma
#    escala que los datos con los que la red aprendio.
# 5. El modelo de TensorFlow analiza la entrada y devuelve las probabilidades de cada numero.
# 6. OpenCV toma la respuesta mas probable y la dibuja de forma elegante sobre el video.

import cv2          # OpenCV: Para la captura, manipulacion y despliegue del video en tiempo real.
import time         # Biblioteca nativa para gestionar pequeñas esperas y pausas de sincronizacion.
import numpy as np  # NumPy: Esencial para formatear las coordenadas en la estructura que la red exige.
import mediapipe as mp # MediaPipe: El motor encargado del seguimiento e identificacion de la mano.
import joblib       # Utilizado para revivir el escalador matematico que guardamos en el entrenamiento.
from tensorflow.keras.models import load_model # Funcion especifica para cargar redes neuronales entrenadas.

# CARGAR MODELO Y ESCALADOR (HERRAMIENTAS DE IA)

print("Cargando modelo y configuraciones de inteligencia artificial...")

# Cargamos la red neuronal en su estado exacto de maxima precision (el mejor modelo guardado).
modelo = load_model("modelo_manos/mejor_modelo.h5")

# Recuperamos el escalador. Si intentaramos pasarle los datos a la red sin normalizarlos
# con este mismo escalador, las predicciones serian erraticas y completamente erroneas.
escalador = joblib.load("modelo_manos/escalador.save")

print("Modelo y escalador cargados correctamente.")

# Definimos una lista ordenada que mapea las salidas de la red neuronal.
# El indice 0 corresponde a la prediccion del string "0", el indice 1 al "1", y asi sucesivamente.
clases = ["0", "1", "2", "3", "4", "5"]

# CONFIGURACION DE MEDIAPIPE HANDS

# Inicializamos los modulos de soluciones de MediaPipe para deteccion de extremidades.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils # Herramienta auxiliar para pintar lineas y puntos en los dedos.

# Creamos el objeto rastreador de manos con parametros especificos de optimizacion.
hands = mp_hands.Hands(
    static_image_mode=False,       # False indica que procesamos un video continuado, acelerando el rastreo.
    max_num_hands=1,               # Limitamos el analisis a una sola mano para ahorrar recursos en la Jetson.
    min_detection_confidence=0.7,  # Umbral de confianza del 70% para detectar una mano por primera vez.
    min_tracking_confidence=0.7    # Umbral del 70% para mantener el seguimiento de la mano en movimiento.
)

# CONFIGURACION DE LA TUBERIA DE GSTREAMER

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    """
    Construye la configuracion de GStreamer para delegar la descompresion y
    el escalado de la imagen al hardware de video dedicado de Nvidia.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink max-buffers=1 drop=true sync=false"
    )


# APERTURA Y VALIDACION DE LA CAMARA

print("Abriendo camara del sistema...")

pipeline = gstreamer_pipeline(sensor_id=0)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# Esperamos dos segundos para que el sensor fisico de la camara ajuste su brillo y contraste.
time.sleep(2)

if not cap.isOpened():
    print("Error: No se pudo abrir la camara. Verifica las conexiones de hardware.")
    exit()

print("Camara abierta correctamente. Presiona la tecla ESC en la ventana de video para salir.")

# BUCLE PRINCIPAL DE INFERENCIA EN TIEMPO REAL

while True:

    ret = False
    frame = None

    # Mecanismo de reintento: Las camaras CSI a veces sufren caidas breves de frames.
    # Intentamos leer la camara hasta 5 veces consecutivas antes de dar el frame por perdido.
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.1)

    if not ret:
        print("Advertencia: No se pudieron recuperar los datos de la camara en este ciclo.")
        continue

    # Espejamos la imagen de forma horizontal (eje 1).
    # Esto se hace para que el usuario experimente un efecto de espejo natural al levantar la mano.
    frame = cv2.flip(frame, 1)

    # Convertimos el formato de color de BGR (OpenCV) a RGB (MediaPipe).
    # MediaPipe fue entrenado con imagenes RGB, por lo que este paso es obligatorio para que sea preciso.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesamos la imagen RGB para buscar la estructura de la mano.
    resultado = hands.process(rgb)

    # Si MediaPipe encontro al menos una mano en la escena, entramos al bloque de prediccion.
    if resultado.multi_hand_landmarks:

        for hand_landmarks in resultado.multi_hand_landmarks:

            # Dibujamos en pantalla los 21 puntos rojos y sus conexiones en color verde.
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Creamos una lista vacia para estructurar el vector de caracteristicas (features).
            puntos = []

            # Extraemos las coordenadas de cada uno de los 21 puntos en orden secuencial.
            for lm in hand_landmarks.landmark:
                # lm.x, lm.y y lm.z son valores proporcionales al tamaño de la pantalla.
                # Al usar extend, guardamos los tres valores de corrido: [x1, y1, z1, x2, y2, z2...]
                puntos.extend([lm.x, lm.y, lm.z])

            # Convertimos la lista de Python en un arreglo de NumPy y cambiamos su forma.
            # .reshape(1, -1) transforma los 63 elementos en una fila con 63 columnas (1, 63).
            # Esto es necesario porque Keras siempre espera un lote (batch) de datos, aun si es una sola muestra.
            entrada = np.array(puntos).reshape(1, -1)

            # Transformamos los datos crudos usando el escalador entrenado.
            # Ahora la media de los datos sera 0 y su desviacion estandar sera 1, estabilizando la red.
            entrada = escalador.transform(entrada)

            # Pasamos los datos normalizados a la red neuronal para obtener las probabilidades.
            # verbose=0 apaga los textos de carga internos de TensorFlow en la terminal para no saturarla.
            pred = modelo.predict(entrada, verbose=0)

            # np.argmax encuentra la posicion (indice) del valor mas alto en el vector de salida.
            # Si la salida es [0.05, 0.90, 0.01...], el valor mas alto esta en el indice 1.
            clase_idx = np.argmax(pred)
            
            # Extraemos el valor flotante de la probabilidad mas alta (por ejemplo, 0.90).
            confianza = np.max(pred)

            # Buscamos el nombre de la clase correspondiente al indice ganador.
            numero_predicho = clases[clase_idx]

            # Construimos la cadena de texto con la prediccion y su porcentaje de precision.
            texto = f"Numero: {numero_predicho} ({confianza:.2f})"

            # Escribimos el texto en la parte superior izquierda de la pantalla en color verde.
            cv2.putText(
                frame,
                texto,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Desplegamos el cuadro final procesado en la ventana interactiva.
    cv2.imshow("Reconocimiento de numeros con la mano", frame)

    # Monitoreamos el teclado. 27 representa la tecla Escape (ESC).
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# CIERRE Y LIBERACION DE COMPONENTES

# Apagamos los servicios de forma ordenada para asegurar el reuso del hardware inmediatamente.
cap.release()          # Apaga el flujo de la camara CSI.
cv2.destroyAllWindows() # Elimina la ventana grafica de la pantalla.
hands.close()          # Cierra de forma interna el hilo de ejecucion de MediaPipe.

print("El programa se cerro de manera exitosa y limpia.")
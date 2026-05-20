import cv2
import mediapipe as mp
import csv
import os

# Inicializar utilidades de dibujo y el modelo de manos de MediaPipe
media_pipe_herramienta_dibujo = mp.solutions.drawing_utils
media_pipe_manos = mp.solutions.hands

manos = media_pipe_manos.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Configuración de la captura de video
captura = cv2.VideoCapture(0)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Mensaje de inicio
print("\nIniciando la recolección de datos para entrenamiento.\n")

folder = "dataset_manos"
os.makedirs(folder, exist_ok=True)
documento_raiz = os.path.join(folder, "datos_manos.csv")

# Creación de cabecera si no existe
if not os.path.exists(documento_raiz):
    with open(documento_raiz, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]]
        writer.writerow(header)
    print(f"Archivo creado exitosamente en {documento_raiz}")

# Entrada de usuario
etiqueta = input("Etiqueta para este gesto (ej. 0, 1, 2, 3, 4, 5): ")

# Prints de instrucciones para el usuario
print(f"\nRecolección de datos para el gesto: {etiqueta}\n")
print("Presiona 's' para guardar una muestra del gesto actual.")
print("Presiona 'ESC' para terminar y salir.\n")

contador = 0 # Para llevar la cuenta en los prints

with open(documento_raiz, mode="a", newline="") as f: 
    escribir = csv.writer(f)

    while True:
        exito, fotograma = captura.read()
        if not exito:
            print("No se pudo leer el frame de la cámara.")
            break

        fotograma = cv2.flip(fotograma, 1)
        alto, ancho, _ = fotograma.shape
        fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
        resultados = manos.process(fotograma_rgb)

        key = cv2.waitKey(1) & 0xFF

        if resultados.multi_hand_landmarks:
            for mano in resultados.multi_hand_landmarks:

                # Dibujo del esqueleto
                media_pipe_herramienta_dibujo.draw_landmarks(
                    fotograma, 
                    mano, 
                    media_pipe_manos.HAND_CONNECTIONS,
                    media_pipe_herramienta_dibujo.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    media_pipe_herramienta_dibujo.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
    

                if key == ord('s'):
                    fila = [etiqueta] + [v for lm in mano.landmark for v in (lm.x, lm.y, lm.z)]
                    escribir.writerow(fila)
                    contador += 1

                    # Este es el print que confirma la acción
                    print(f"Muestra #{contador} para '{etiqueta}' guardada.")
                    print("Muestra guardada exitosamente en el archivo CSV.")

        # Texto en pantalla
        cv2.putText(
            fotograma, 
            f"Gesto: {etiqueta} -> Muestras: {contador}",
            (10, 30),   
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )

        cv2.imshow("Recoleccion de Datos para Entrenamiento", fotograma)

        if key == 27:  # Tecla ESC
            print(f"Sesión finalizada por el usuario. Total de muestras capturadas: {contador}")
            break

# Liberación de recursos
captura.release()
cv2.destroyAllWindows()
manos.close()
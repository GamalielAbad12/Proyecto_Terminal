# Este código utiliza la biblioteca MediaPipe para detectar y dibujar 
# las manos en tiempo real utilizando la cámara web.

# Asegúrate de tener instalada la biblioteca MediaPipe y OpenCV 
# para ejecutar este código correctamente.

# Puedes instalar MediaPipe con: pip install mediapipe
# Puedes instalar OpenCV con: pip install opencv-python

import cv2
import mediapipe as mp

# Inicializar utilidades de dibujo y el modelo de manos
media_pipe_herramientas_dibujar = mp.solutions.drawing_utils
media_pipe_manos = mp.solutions.hands

# Configuración del modelo de manos
captura = cv2.VideoCapture(0)

#Configuración de la cámara (resolución)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Inicializar el objeto de detección de manos
manos =  media_pipe_manos.Hands(
    static_image_mode=False, # Tratar como flujo de video
    max_num_hands=2, # Máximo de manos a detectar
    min_detection_confidence=0.5, # Umbral de confianza para detección
    min_tracking_confidence=0.5 # Umbral de confianza para seguimiento
)

print("Presiona 'ESC' para salir")

while True:

    # Capturar un frame de la cámara
    si_captura, imagen = captura.read()
   
    # Si no se pudo capturar, salir del bucle
    if not si_captura:
        print("No se pudo capturar la imagen.")
        break

    # Para voltear la imagen (modo espejo)
    imagen = cv2.flip(imagen, 1) 

    # Convertir la imagen a RGB (MediaPipe trabaja en RGB)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) 
    
    # Procesar la imagen para detectar manos
    resultados = manos.process(imagen_rgb)

    if resultados.multi_hand_landmarks:
        for mano in resultados.multi_hand_landmarks:
            # Dibujar los puntos y conexiones de la mano en la imagen
            media_pipe_herramientas_dibujar.draw_landmarks(
                imagen, #Necesitamos la imagen para dibujar sobre ella
                mano, #Necesitamos los landmarks de la mano detectada
                media_pipe_manos.HAND_CONNECTIONS, #Nwecesitamos las conexiones entre los puntos para dibujarlas

                # Color de los puntos (Landmarks)
                media_pipe_herramientas_dibujar.DrawingSpec(color=(255, 245, 245), thickness=3, circle_radius=2), 
                # Color de las líneas (Conexiones)
                media_pipe_herramientas_dibujar.DrawingSpec(color=(201, 16, 198), thickness=2))
            
    # Mostrar la imagen con las detecciones
    cv2.imshow("Detección de manos basica", imagen)

    # Salir si se presiona la tecla 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

    #El bucle se repetirá continuamente, capturando frames y procesándolos hasta que el usuario presione 'ESC' para salir.


# Liberar recursos

# Liberar la captura de video
captura.release()

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()

# Cerrar el objeto de detección de manos
manos.close()
# Este código utiliza la biblioteca MediaPipe para detectar la punta del dedo índice
# en tiempo real utilizando la cámara web.

# Asegúrate de tener instalada la biblioteca MediaPipe y OpenCV 
# para ejecutar este código correctamente.
# pip install mediapipe opencv-python

import cv2
import mediapipe as mp

# Inicializar utilidades de dibujo y el modelo de manos de MediaPipe
media_pipe_herramientas_dibujar = mp.solutions.drawing_utils
media_pipe_manos = mp.solutions.hands

# Configuración de la captura de video (Cámara por defecto)
captura = cv2.VideoCapture(0)

# Configuración de la resolución de la cámara
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Inicializar el objeto de detección de manos
# min_detection_confidence: Confianza mínima para detectar la mano (0.5 = 50%)
# min_tracking_confidence: Confianza mínima para seguir la mano después de detectarla (0.5 = 50%)
manos = media_pipe_manos.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

print("Presiona 'ESC' para salir")

while True:
    # Capturar un frame de la cámara
    si_captura, imagen = captura.read()
   
    # Si la cámara falla, salir del bucle
    if not si_captura:
        print("No se pudo capturar la imagen.")
        break

    #Modo espejo para que sea intuitivo mover la mano
    imagen = cv2.flip(imagen, 1) 

    #MediaPipe requiere imágenes en formato RGB
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) 
    
    #Procesar la imagen para encontrar los 21 puntos de la mano
    resultados = manos.process(imagen_rgb)

    #Obtener las dimensiones de la imagen (Importante: Alto primero, luego Ancho)
    alto, ancho, _ = imagen.shape

    #Si se detecta al menos una mano, trabajar con sus puntos
    if resultados.multi_hand_landmarks:
        for mano in resultados.multi_hand_landmarks:
            
            # Seleccionamos el punto 8: Punta del dedo índice
            punto_indice = mano.landmark[8] 
            
            # Convertimos las coordenadas relativas (0 a 1) en píxeles reales
            punto_indice_x = int(punto_indice.x * ancho)
            punto_indice_y = int(punto_indice.y * alto)

            # Dibujamos un círculo verde sobre la punta del dedo índice
            # (imagen, coordenadas, radio, color BGR, relleno)
            cv2.circle(imagen, (punto_indice_x, punto_indice_y), 10, (0, 255, 0), cv2.FILLED)

            # Imprimimos las coordenadas
            # Z será negativo si el dedo está más cerca de la cámara que la muñeca.
            print(f"Índice detectado -> X: {punto_indice.x:.4f}, Y: {punto_indice.y:.4f}, Z: {punto_indice.z:.4f}")

    #Mostrar la ventana con el punto dibujado
    cv2.imshow("Detección de Dedo Índice", imagen)

    # Salir si el usuario presiona la tecla 'ESC' (ASCII 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Apagamos la cámara
captura.release()
# Cerramos las ventanas gráficas
cv2.destroyAllWindows()
# Liberamos la memoria del modelo de manos
manos.close()
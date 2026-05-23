# Este programa fue desarrollado con el propósito de gestionar y visualizar de manera
# simultánea el flujo de video de dos cámaras conectadas a una placa Nvidia Jetson.
# Está diseñado específicamente para sensores como el IMX219 (comunes en aplicaciones
# de visión estéreo o robótica), los cuales requieren una configuración especial de hardware.

# Para lograrlo, el script realiza tareas críticas de bajo nivel:
# 1. Reinicia el servicio del sistema encargado de controlar las cámaras en la Jetson, 
#    lo que previene bloqueos comunes por recursos mal liberados en ejecuciones previas.
# 2. Configura una tubería (pipeline) de GStreamer para delegar el procesamiento pesado de 
#    video (como el cambio de formato de color) directamente al chip de Nvidia.
# 3. Captura los fotogramas de ambas cámaras en tiempo real, los redimensiona para asegurar
#    su compatibilidad y los concatena horizontalmente en una sola ventana visual.

# Este flujo es fundamental en proyectos de visión artificial donde se necesita calibrar
# dos cámaras al mismo tiempo o capturar imágenes coordinadas para calcular profundidad.

import os      # Permite interactuar con el sistema operativo para ejecutar comandos de terminal.
import cv2     # OpenCV: Biblioteca principal para capturar, procesar y mostrar el video.
import numpy as np # NumPy: Se utiliza aquí para manipular las matrices de píxeles y unir los videos.
import time    # Proporciona funciones de temporización para pausar el script cuando sea necesario.

# REINICIO DEL DEMONIO DE ARGUS (NVIDIA JETSON)

# Las placas Nvidia Jetson utilizan un servicio del sistema llamado "nvargus-daemon" 
# para comunicarse con los sensores de la cámara. Con mucha frecuencia, si un programa 
# se cierra de forma inesperada, este servicio se queda congelado o bloquea el acceso.
# Ejecutar este reinicio al principio asegura que el hardware comience desde un estado limpio.

# Enviamos un comando con permisos de administrador para reiniciar el servicio de la cámara.
os.system("sudo systemctl restart nvargus-daemon")

# Pausamos la ejecución durante 3 segundos. Esto le da tiempo suficiente al hardware 
# y al sistema operativo para inicializarse por completo antes de intentar abrir los lentes.
time.sleep(3)


def gstreamer_pipeline(sensor_id=0, width=1280, height=720, framerate=30):
    """
    Construye una cadena de texto que configura GStreamer.
    GStreamer es un framework que permite transferir el video directamente a través 
    del hardware de Nvidia, optimizando drásticamente el uso del procesador (CPU).
    """
    return (
        # nvarguscamerasrc: Indica que usaremos el controlador de cámaras de Nvidia para el sensor especificado.
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        
        # NVMM (Nvidia Memory Management): Indica que el video se guardará directamente en la memoria 
        # optimizada de video. Aquí definimos la resolución (1280x720), los FPS (30) y el formato bruto (NV12).
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={framerate}/1, format=NV12 ! "
        
        # nvvidconv: Convertidor de video por hardware de Nvidia. Transforma el formato bruto de la cámara.
        "nvvidconv ! "
        
        # Pasamos el video a un formato de memoria estándar convertible por el procesador, usando BGRx.
        "video/x-raw, format=BGRx ! "
        
        # videoconvert: Convierte el formato BGRx (que incluye un canal vacío) al formato BGR estándar.
        "videoconvert ! "
        
        # Especificamos que OpenCV recibirá el formato BGR tradicional (Azul, Verde, Rojo).
        "video/x-raw, format=BGR ! "
        
        # appsink: El punto de salida que conecta GStreamer con nuestro código de Python.
        # drop=true asegura que si el código se retrasa, se descarten los fotogramas viejos para evitar lag.
        "appsink sync=false drop=true"
    )

# INICIALIZACIÓN DE LAS CÁMARAS

# Usamos el objeto VideoCapture de OpenCV, pero en lugar de pasarle un número entero 
# (como en una computadora normal), le pasamos la tubería de GStreamer que creamos arriba.

print("Abriendo camara 0...")
cap0 = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)

print("Abriendo camara 1...")
cap1 = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

# Validamos si la cámara 0 se abrió correctamente. Si no es así, detenemos el programa.
if not cap0.isOpened():
    print("Error: No se pudo abrir la camara 0")
    exit()

# Validamos si la cámara 1 se abrió correctamente. Si falla, cerramos la cámara 0 para no dejar hilos abiertos.
if not cap1.isOpened():
    print("Error: No se pudo abrir la camara 1")
    cap0.release()
    exit()

print("Ambas camaras abiertas correctamente. Presiona la tecla ESC para salir.")

# BUCLE PRINCIPAL DE CAPTURA Y PROCESAMIENTO

while True:
    # Capturamos el fotograma actual de ambas cámaras de manera simultánea.
    # 'ret' es un valor booleano (True/False) que indica si la captura fue exitosa.
    # 'frame' contiene la matriz de píxeles de la imagen.
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    # Si alguna de las dos cámaras falla al entregar la imagen, rompemos el ciclo.
    if not ret0:
        print("Advertencia: Error leyendo los datos de la camara 0")
        break

    if not ret1:
        print("Advertencia: Error leyendo los datos de la camara 1")
        break

    # Mecanismo de seguridad: Para poder juntar dos imágenes en una sola ventana, 
    # ambas deben tener exactamente las mismas dimensiones. 
    # Forzamos a que el fotograma de la cámara 1 adopte el ancho y alto del fotograma de la cámara 0.
    frame1 = cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))

    # hstack (Horizontal Stack): Función de NumPy que une dos matrices de forma horizontal.
    # Coloca la imagen de la cámara 0 a la izquierda y la de la cámara 1 a la derecha.
    combined = np.hstack((frame0, frame1))

    # Muestra el resultado de la unión de las pantallas en una ventana interactiva.
    cv2.imshow("Vista doble de camaras imx219", combined)

    # Espera 1 milisegundo para detectar si el usuario presiona una tecla.
    # El valor 27 corresponde al código ASCII de la tecla ESC (Escape).
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# LIMPIEZA DE RECURSOS
# Al terminar el programa, es fundamental liberar las cámaras y cerrar las ventanas.
# Esto previene que el hardware se quede bloqueado para futuras ejecuciones.

cap0.release()          # Libera el enlace con el sensor 0.
cap1.release()          # Libera el enlace con el sensor 1.
cv2.destroyAllWindows() # Cierra todas las ventanas gráficas creadas por OpenCV.

print("El programa se cerro correctamente.")
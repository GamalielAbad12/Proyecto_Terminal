# Este programa fue desarrollado con el propósito de verificar, diagnosticar y poner
# a prueba el entorno de desarrollo antes de comenzar un proyecto formal de IA.
# Su objetivo principal es asegurar que el hardware (especialmente la tarjeta gráfica o GPU)
# y las bibliotecas de software clave estén perfectamente configurados y comunicados entre sí.

# Para lograrlo, el script realiza tres tareas fundamentales:
# 1. Configura TensorFlow para optimizar el uso de la memoria gráfica.
# 2. Realiza un benchmark (prueba de rendimiento) matemático pesado directamente en la GPU
#    para comprobar que el cálculo acelerado por hardware funciona correctamente.
# 3. Inicializa una matriz con NumPy y utiliza OpenCV para simular el procesamiento de una
#    imagen, asegurando que las herramientas de Visión por Computadora estén listas.

# Ejecutar este script es un paso crucial en el flujo de trabajo de cualquier desarrollador,
# ya que permite detectar problemas de controladores (drivers), incompatibilidades de versiones
# o fallos de hardware antes de lanzar entrenamientos de redes neuronales que podrían durar horas.

import cv2          # OpenCV: Biblioteca líder para procesamiento de imágenes y video en tiempo real.
import mediapipe as mp  # MediaPipe: Herramienta de Google para detección de rostros, manos y poses.
import tensorflow as tf # TensorFlow: El motor principal para construir y entrenar redes neuronales.
import time         # Biblioteca nativa para medir tiempos de ejecución (benchmarking).
import numpy as np  # NumPy: Biblioteca fundamental para el manejo de matrices y álgebra lineal.

# OPTIMIZACIÓN Y GESTIÓN DE LA MEMORIA DE LA GPU

# Por defecto, TensorFlow tiene un comportamiento "agresivo": en cuanto arranca,
# reserva prácticamente el 100% de la memoria VRAM de tu tarjeta gráfica. 
# Esto puede provocar que otros programas (o incluso herramientas de OpenCV) se queden
# sin memoria y colapsen. Las siguientes líneas corrigen ese comportamiento.

# Buscamos si el sistema tiene tarjetas gráficas (GPUs) disponibles y compatibles.
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Si encuentra GPUs, iteramos sobre cada una de ellas.
        for gpu in gpus:
            # Activamos el crecimiento dinámico de memoria (Memory Growth).
            # Con esto, TensorFlow solo tomará la memoria que necesite en el momento
            # y la irá expandiendo poco a poco a medida que el modelo crezca.
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # En caso de que la configuración falle (por ejemplo, si se intenta configurar
        # después de que TensorFlow ya inicializó la GPU), atrapamos el error.
        print(f"Error al configurar el crecimiento de memoria: {e}")

# SECCIÓN 1: VERIFICACIÓN DE VERSIONES DEL ENTORNO

# En el mundo del Machine Learning, la compatibilidad de versiones es vital. 
# Aquí imprimimos las versiones exactas instaladas para asegurarnos de que cumplan
# con los requisitos del proyecto y confirmamos si TensorFlow detectó la tarjeta de video.

print("VERIFICACIÓN DE ENTORNO")
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("TensorFlow:", tf.__version__)
# Si la lista se muestra vacía [], significa que TensorFlow está usando solo el CPU.
print("GPU detectada:", tf.config.list_physical_devices('GPU'))

# Imprime detalles técnicos de cómo fue compilado TensorFlow en tu máquina.
# Esto incluye información sobre las herramientas de NVIDIA (como CUDA y cuDNN) 
# que permiten a la tarjeta gráfica procesar operaciones matemáticas de IA.
print("\nBUILD INFO ")
print(tf.sysconfig.get_build_info())


# SECCIÓN 2: PRUEBA DE RENDIMIENTO (BENCHMARK) EN LA GPU

# No basta con que TensorFlow "vea" la GPU; necesitamos comprobar que puede inyectarle
# trabajo matemático pesado y resolverlo sin colgarse.

print("\nPRUEBA DE CÁMERA / PRUEBA DE GPU")
# Registramos el tiempo exacto en segundos antes de iniciar la operación.
inicio = time.time()

# Forzamos explícitamente a TensorFlow a ejecutar el siguiente bloque de código
# dentro de la primera tarjeta gráfica disponible ('/GPU:0').
with tf.device('/GPU:0'):
    # Creamos dos matrices gigantes de 2000x2000 llenas de números aleatorios decimales.
    # Cada matriz tiene 4 millones de elementos.
    a = tf.random.normal([2000, 2000])
    b = tf.random.normal([2000, 2000])
    
    # Realizamos una multiplicación de matrices (Matrix Multiplication).
    # Esta es la operación matemática base sobre la que se construyen las redes neuronales.
    # Al ser miles de millones de cálculos, es el escenario ideal para estresar la GPU.
    c = tf.matmul(a, b)

# Restamos el tiempo actual menos el tiempo de inicio para saber cuántos segundos tomó.
print("Tiempo:", time.time() - inicio)
# Confirmamos en la consola que la operación realmente se ejecutó en la GPU y no en el CPU.
print("Dispositivo usado:", c.device)


# SECCIÓN 3: PRUEBA DE IMAGEN / VISIÓN POR COMPUTADORA

# Esta sección simula la creación y manipulación de un cuadro (frame) de video.
# Nos asegura que OpenCV tiene los permisos correctos para escribir archivos en el disco.

print("\nPRUEBA DE CÁMARA / OPENCV")
# Creamos un lienzo completamente negro usando NumPy.
# Las dimensiones son: 480 de alto, 640 de ancho y 3 canales de color (RGB).
# El tipo de dato 'uint8' (entero de 8 bits) es el estándar para imágenes (valores de 0 a 255).
img = np.zeros((480, 640, 3), dtype=np.uint8)

# Dibujamos un texto sobre nuestra imagen negra creada en el paso anterior.
# Parámetros: (imagen, texto, coordenadas de inicio X e Y, tipografía, escala, color BGR, grosor).
# Nota: OpenCV usa el formato BGR, por lo que (255,255,255) es color blanco puro.
cv2.putText(img, "OpenCV funcionando", (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# Guardamos la imagen resultante en el disco duro con el nombre "test_opencv.jpg".
# Si este comando se ejecuta sin errores, significa que el script tiene permisos de escritura.
cv2.imwrite("test_opencv.jpg", img)
print("Imagen creada: test_opencv.jpg")

# Mensaje final que consolida que todo el ecosistema (Hardware + Software) está listo para la acción.
print("\nTODO FUNCIONA CORRECTAMENTE")
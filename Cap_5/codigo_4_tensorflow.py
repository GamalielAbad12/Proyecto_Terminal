# Este programa fue desarrollado con el propósito de aprender a construir una red neuronal
# convolucional (CNN) capaz de detectar si en una imagen aparece una mano o no.
# Para ello, utilizamos un conjunto de imágenes organizadas en carpetas, donde cada carpeta
# representa una categoría distinta. En este caso tenemos dos clases:
# una carpeta llamada "con mano" y otra llamada "sin mano".
#
# El modelo aprenderá observando muchos ejemplos de ambas categorías hasta identificar
# patrones visuales que le permitan clasificar imágenes nuevas que nunca haya visto.
#
# Para lograrlo utilizamos TensorFlow como biblioteca principal de inteligencia artificial,
# NumPy para trabajar con arreglos numéricos, y herramientas auxiliares de TensorFlow
# para cargar, procesar y normalizar imágenes antes de entrenar la red.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

# Leyendo y estudiando algunos modelos, vamos a utilizar imágenes para el procesamiento.
# Estas imágenes están organizadas en lo que llamamos un *dataset*. En nuestro caso, el dataset contiene dos carpetas:
# una titulada "con mano" y otra llamada "sin mano".

ruta_dataset = 'dataset_img_manos/'

# TensorFlow proporciona la herramienta ImageDataGenerator, la cual administra, reescala y normaliza las imágenes.
# El argumento `validation_split=0.2` indica que el 20% de las imágenes se usarán para validación y el 80% restante para entrenamiento.
# El argumento `rescale=1./255` normaliza los píxeles de las imágenes. Los valores RGB van de 0 a 255, 
# por lo que al dividir por 255 obtenemos un rango entre 0 y 1. Por ejemplo, si un píxel tiene un valor de 124, se transforma en 124/255 ≈ 0.48.

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Ahora, creamos los datos de entrenamiento utilizando el 80% de las imágenes.
# `target_size=(128, 128)` reescala las imágenes a 128x128 píxeles para reducir la cantidad de neuronas de entrada.
# `batch_size=32` permite entrenar el modelo en bloques de 32 imágenes a la vez, optimizando el uso de memoria.
# `class_mode='binary'` indica que solo hay dos clases (con mano y sin mano), y asignará automáticamente 0 o 1 a cada imagen.
# En caso de que existan más carpetas (más clases), se puede usar `class_mode='categorical'`, lo cual asignará un número distinto a cada carpeta.
# Finalmente, `subset='training'` indica que estas imágenes serán usadas para el entrenamiento.

train_data = datagen.flow_from_directory(
    ruta_dataset,
    target_size=(128, 128),  
    batch_size=20,
    class_mode='binary',
    subset='training'
)

# Esto nos permite verificar a qué etiqueta (0 o 1) corresponde cada carpeta.
print(train_data.class_indices)

# Repetimos el mismo procedimiento anterior, pero ahora para obtener los datos de validación.
# `subset='validation'` indica que este conjunto de imágenes será usado para validar el desempeño del modelo.

val_data = datagen.flow_from_directory(
    ruta_dataset,
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary',
    subset='validation'
)

# Aquí comenzamos con la creación del modelo.
# Utilizamos un modelo secuencial, el cual es una pila de capas lineales.
# Aunque existen otras arquitecturas, por simplicidad y aprendizaje trabajamos con esta.

modelo = tf.keras.Sequential([

    # Primera capa convolucional + MaxPooling
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Segunda capa convolucional + MaxPooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Tercera capa convolucional + MaxPooling
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Apagael 30% de neuronas
    tf.keras.layers.Dropout(0.3),

    # Prueba actual 
    tf.keras.layers.Flatten(),

    # Capa densa intermedia para clasificación, muy profunda.
    tf.keras.layers.Dense(128, activation='relu'),

    # Otro Dropouo,apaga el 30% de las neuronas
    tf.keras.layers.Dropout(0.3),

    # Capa de salida binaria
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilamos el modelo usando el optimizador Adam y una función de pérdida binaria (dos clases).
modelo.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Entrenamos el modelo usando los datos de entrenamiento y validación durante 10 épocas.
modelo.fit(
    train_data,
    validation_data=val_data,
    epochs=50
)

print("Entrenamiento terminado")

# Probamos el modelo con una imagen externa.
img = image.load_img('dataset_img_manos/test_1.jpg', target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediccion = modelo.predict(img_array)
print("Hay una mano" if prediccion[0][0] > 0.5 else "No hay mano")

# Segunda prueba con otra imagen.
img = image.load_img('test_2.jpg', target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediccion = modelo.predict(img_array)
print("Hay una mano" if prediccion[0][0] > 0.5 else "No hay mano")

# Finalmente, guardamos el modelo entrenado para poder reutilizarlo más adelante sin tener que volver a entrenarlo.
modelo.save('modelo_manos_v2.h5')
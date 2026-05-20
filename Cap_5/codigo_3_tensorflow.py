# Este programa fue realizado con el propósito de aprender cómo construir una red neuronal
# utilizando la biblioteca TensorFlow. El objetivo es que la red aprenda el comportamiento
# de una compuerta lógica AND a partir de ejemplos de entrada y salida.

# Comenzamos importando TensorFlow y NumPy. TensorFlow será nuestra herramienta principal
# para crear y entrenar la red, y NumPy nos ayudará a manejar los datos.
import tensorflow as tf
import numpy as np

# Definimos las entradas para la compuerta AND. Estas son las combinaciones posibles de 0 y 1.
# Son vectores de dos componentes, ya que la compuerta AND toma dos valores de entrada.
entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]], dtype=np.float32)

# Estas son las salidas esperadas para cada combinación de entrada.
# Según la tabla de verdad de la compuerta AND, sólo (1 AND 1) da como resultado 1.
salidas = np.array([[0], [0], [0], [1]], dtype=np.float32)

# Creamos el modelo de red neuronal. Usamos el tipo Sequential, que permite apilar capas de forma lineal.
# Aquí solo tenemos una única capa (una neurona) con activación sigmoide.
# La capa espera vectores de entrada de tamaño 2 (porque hay dos entradas).
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,))
])

# Compilamos el modelo. Esto significa configurarlo para el entrenamiento.
# Usamos:
# - El optimizador 'adam', que ajusta los pesos automáticamente de forma eficiente.
# - La función de pérdida 'binary_crossentropy', ideal para problemas de clasificación binaria.
# - Y como métrica adicional, la exactitud (accuracy).
modelo.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Entrenamos el modelo. Le damos:
# - Las entradas y salidas que definimos antes.
# - El número de épocas (repeticiones sobre todos los datos).
# - verbose=0 desactiva los mensajes para que no se vea tanto texto.
# Podemos cambiar a 1 si queremos ver el progreso.
modelo.fit(entradas, salidas, epochs=9000, verbose=0)

# Una vez entrenado el modelo, lo probamos con las mismas entradas para ver si aprendió bien.
# Usamos un ciclo para predecir una por una y ver los resultados.

print("Resultados luego del entrenamiento:\n")
for entrada in entradas:
    # La predicción devuelve un valor entre 0 y 1 (por la sigmoide).
    # Lo redondeamos para que se parezca a una salida binaria (0 o 1).
    prediccion = modelo.predict(np.array([entrada]), verbose=0)
    print(f"{entrada} => {prediccion[0][0]:.4f} (≈ {round(prediccion[0][0])})")

# Como resultado final, deberíamos obtener algo cercano a:
# [0 0] => ~0
# [0 1] => ~0
# [1 0] => ~0
# [1 1] => ~1

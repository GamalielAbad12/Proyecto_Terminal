# Este programa fue realizado con la intención de aprender sobre redes neuronales.
# Utilizando un ejemplo relacionado con la compuerta lógica AND, este código es meramente educativo.
# Se intentará dar una explicación detallada del funcionamiento de una red neuronal simple.

# Dado que deseamos una red neuronal que, dadas unas entradas, nos dé ciertas salidas,
# tenemos que para la compuerta AND:

#  - 1 AND 1 = 1
#  - 1 AND 0 = 0
#  - 0 AND 1 = 0
#  - 0 AND 0 = 0

# Para iniciar con nuestra red neuronal necesitamos las entradas. Usaremos la librería 
# de Python llamada NumPy para realizarlo.

import numpy as np  # Usamos un alias estándar

# Definimos el arreglo de entradas.
# En este caso, agregamos una tercera columna con valor constante 1.
# Esta columna representa el BIAS (sesgo), el cual permite a la red neuronal
# ajustar la frontera de decisión y no depender únicamente del origen.
entradas = np.array([
    [0, 0, 1],  # Bias = 1
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Como función de activación utilizaremos la función sigmoide, 
# la cual es recomendada para empezar por su simplicidad.
def funcion_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Implementamos la derivada de la sigmoide de forma simplificada.
# Al recibir el valor ya activado (la salida), la fórmula es:
# f'(x) = x * (1 - x)
def derivada_sigmoid(x):
    return x * (1 - x)

# Definimos las salidas esperadas según el orden de las entradas.
salidas_esperadas = np.array([[0], [0], [0], [1]])

# Vamos ahora con uno de los aspectos más importantes de las redes neuronales: los pesos.
# Para poder reproducir los mismos resultados cada vez, fijamos una semilla aleatoria.
np.random.seed(11)

# Inicializamos los pesos con valores aleatorios entre [-1, 1].
# Ahora tenemos 3 pesos (dos entradas + bias).
pesos = 2 * np.random.rand(3, 1) - 1

# Definimos la tasa de aprendizaje.
tasa_aprendizaje = 0.1

# Definimos el número de épocas.
for epoca in range(100000):

    # Calculamos el producto punto entre las entradas y los pesos.
    # Esto incluye automáticamente el efecto del bias.
    entrada_dot_pesos = np.dot(entradas, pesos)

    # Aplicamos la función de activación para obtener la salida de la red.
    salida = funcion_sigmoid(entrada_dot_pesos)

    # Calculamos el error: salida esperada menos salida actual.
    error = salidas_esperadas - salida

    # Calculamos el ajuste utilizando la derivada de la función sigmoide.
    ajustes = error * derivada_sigmoid(salida)

    # Ajustamos los pesos.
    # Notar que el bias también se ajusta, ya que forma parte del vector de pesos.
    pesos += np.dot(entradas.T, ajustes) * tasa_aprendizaje

    # (Opcional) Monitoreo del error cada cierto número de épocas.
    if epoca % 20000 == 0:
        print(f"Error en época {epoca}: {np.mean(np.abs(error)):.4f}")

# Teóricamente, nuestra red neuronal ya debería estar entrenada.
# Probamos ahora con las mismas entradas que usamos para entrenar.
print("\nPesos entrenados:\n", pesos)

print("\nSalida luego del entrenamiento:")
predicciones = funcion_sigmoid(np.dot(entradas, pesos))
print(predicciones)

# Verificación clara de resultados
print("\nResultados finales:")
for i in range(len(entradas)):
    print(f"Entrada: {entradas[i][:2]} -> Predicción: {predicciones[i][0]:.4f}")
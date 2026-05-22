# Este programa fue desarrollado con el propósito de entrenar una red neuronal capaz
# de reconocer números hechos con la mano utilizando landmarks (puntos de referencia)
# obtenidos de la detección de manos. Cada muestra del dataset contiene las coordenadas
# X, Y y Z de 21 puntos de la mano, y la red aprenderá a asociar esas posiciones con
# un número específico.

# Para lograrlo, utilizamos TensorFlow como biblioteca principal para construir
# y entrenar la red neuronal. También empleamos NumPy para trabajar con arreglos
# numéricos, Pandas para leer el dataset desde un archivo CSV, Scikit-learn para
# dividir y normalizar los datos, y Joblib para guardar herramientas auxiliares
# como el escalador que necesitaremos al hacer predicciones en el futuro.

# A lo largo del programa se realizará un proceso completo de aprendizaje automático:
# primero se cargan y preparan los datos, luego se construye la arquitectura de la red,
# después se entrena con ejemplos, se evalúa con datos nunca vistos, y finalmente
# se guarda el modelo entrenado para poder reutilizarlo más adelante sin necesidad
# de volver a entrenarlo.

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib  # Librería para guardar objetos en disco (como el escalador)

# Ruta donde se encuentra nuestro dataset.
# El archivo CSV contiene:
# label, x1, y1, z1, x2, y2, z2 ... hasta completar 21 puntos de la mano.
directorio_csv = "dataset_manos/datos_manos.csv"

# Carpeta donde se guardará el modelo entrenado.
directorio_modelo = "modelo_manos"

# Nombre del archivo final del modelo.
nombre_del_modelo = "modelo_numeros_manos"

# La semilla hace que el entrenamiento sea reproducible.
# Es decir, si volvemos a entrenar, los resultados serán muy parecidos.
semilla_random = 42

# Cantidad de muestras que la red procesa antes de ajustar sus pesos.
# Si tenemos 1000 muestras y batch_size=32, entrenará en bloques.
tamano_lote = 32

# Número máximo de veces que el modelo verá el dataset completo.
epocas = 100


def cargar_dataset(ruta_csv):
    """
    Carga el archivo CSV y separa:
    X = coordenadas de la mano
    y = número representado (label)
    """

    # Leemos el archivo CSV usando pandas.
    df = pd.read_csv(ruta_csv)

    # Verificamos que exista la columna label.
    # Esta columna contiene la respuesta correcta (0,1,2,3,4,5...)
    if "label" not in df.columns:
        raise ValueError("El CSV debe tener una columna 'label' como primera columna.")

    # Extraemos las etiquetas (variable objetivo)
    # Ejemplo:
    # [0, 1, 2, 5, 3, 4]
    etiquetas = df["label"].astype(int).to_numpy()

    # Eliminamos la columna label y nos quedamos solo con las coordenadas.
    # Cada fila tendrá 63 valores:
    # 21 puntos × (X,Y,Z)
    entradas = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    # Validamos que sí tengamos las 63 coordenadas esperadas.
    if entradas.shape[1] != 63:
        print(f"Advertencia: se detectaron {entradas.shape[1]} coordenadas (se esperaban 63).")

    return entradas, etiquetas


def preparar_datos(entradas, etiquetas, tamano_prueba=0.15, tamano_val=0.15, semilla=semilla_random):
    """
    Divide el dataset en:
    - Entrenamiento
    - Validación
    - Prueba

    Y además normaliza los datos.
    """

    # Mezclamos los datos.
    # Esto evita que el orden en que fueron capturados afecte el aprendizaje.
    entradas, etiquetas = shuffle(entradas, etiquetas, random_state=semilla)

    # Primera división:
    # 70% entrenamiento
    # 30% temporal (de aquí luego saldrá validación y prueba)
    (X_entrenamiento, X_temporal,
     y_entrenamiento, y_temporal) = train_test_split(
        entradas, etiquetas,
        test_size=(tamano_prueba + tamano_val),
        random_state=semilla,
        stratify=etiquetas
    )

    # Calculamos cuánto del temporal será validación.
    porcion_val = tamano_val / (tamano_prueba + tamano_val)

    # Segunda división:
    # Del grupo temporal obtenemos:
    # Validación y Prueba
    (X_validacion, X_prueba,
     y_validacion, y_prueba) = train_test_split(
        X_temporal, y_temporal,
        test_size=(1.0 - porcion_val),
        random_state=semilla,
        stratify=y_temporal
    )

    # StandardScaler convierte los datos para que:
    # media = 0
    # desviación estándar = 1
    #
    # Ejemplo:
    # Si una coordenada vale 150 y otra 0.03,
    # el escalador las lleva a un rango comparable.
    #
    # Esto ayuda mucho al entrenamiento de redes neuronales.
    escalador = StandardScaler()

    X_entrenamiento_esc = escalador.fit_transform(X_entrenamiento)

    # Importante:
    # Aquí NO usamos fit_transform porque el escalador
    # debe aprender SOLO del entrenamiento.
    X_validacion_esc = escalador.transform(X_validacion)
    X_prueba_esc = escalador.transform(X_prueba)

    return (
        (X_entrenamiento_esc, y_entrenamiento),
        (X_validacion_esc, y_validacion),
        (X_prueba_esc, y_prueba),
        escalador
    )


def construccion_modelo_neuronal_denso(dimension_entrada, num_clases=6):
    """
    Crea la red neuronal profunda para clasificar números hechos con la mano.
    """

    modelo = models.Sequential([

        # Recibe 63 valores:
        # 21 landmarks × 3 coordenadas (X,Y,Z)
        layers.Input(shape=(dimension_entrada,)),

        # 256 neuronas:
        # Cada neurona aprende patrones en las coordenadas.
        layers.Dense(256, activation="relu"),

        # Normaliza activaciones internas para estabilizar entrenamiento.
        layers.BatchNormalization(),

        # Apaga aleatoriamente el 25% de neuronas en entrenamiento.
        # Esto evita que la red memorice (overfitting).
        layers.Dropout(0.25),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation="relu"),

        # Una neurona por clase.
        # Si tenemos números 0 a 5:
        # habrá 6 neuronas.
        #
        # Softmax convierte la salida en probabilidades.
        # Ejemplo:
        # [0.01, 0.03, 0.90, 0.02, 0.01, 0.03]
        # Aquí predice clase 2.
        layers.Dense(num_clases, activation="softmax")
    ])

    # Adam = optimizador que ajusta pesos eficientemente.
    #
    # sparse_categorical_crossentropy:
    # Se usa cuando las etiquetas son enteros:
    # 0,1,2,3...
    #
    # accuracy:
    # mide porcentaje de aciertos.
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


def main():

    # Hacemos reproducibles los experimentos.
    tf.random.set_seed(semilla_random)
    np.random.seed(semilla_random)

    # Verificamos que el CSV exista.
    if not os.path.exists(directorio_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {directorio_csv}")

    print("Cargando y procesando datos...")

    # Cargamos dataset.
    X, y = cargar_dataset(directorio_csv)

    # Detectamos automáticamente cuántas clases existen.
    # Ejemplo:
    # Si y tiene [0,1,2,3,4,5] => 6 clases
    clases_detectadas = len(np.unique(y))

    # Preparamos los datos
    (X_entreno, y_entreno), (X_val, y_val), (X_test, y_test), escalador = preparar_datos(X, y)

    # Construimos la red neuronal
    modelo = construccion_modelo_neuronal_denso(
        X_entreno.shape[1],
        num_clases=clases_detectadas
    )

    # Muestra la arquitectura en consola.
    modelo.summary()

    # Son funciones especiales que actúan durante el entrenamiento.

    os.makedirs(directorio_modelo, exist_ok=True)

    ruta_mejor_modelo = os.path.join(directorio_modelo, "mejor_modelo.h5")

    # Si el modelo deja de mejorar durante 10 épocas:
    # detiene entrenamiento.
    detencion_temprana = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Guarda automáticamente el mejor modelo.
    guardar_mejor = callbacks.ModelCheckpoint(
        ruta_mejor_modelo,
        monitor="val_loss",
        save_best_only=True
    )

    # Si el modelo se estanca:
    # reduce el learning rate para afinar el aprendizaje.
    ajustar_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    historia = modelo.fit(
        X_entreno, y_entreno,
        validation_data=(X_val, y_val),
        epochs=epocas,
        batch_size=tamano_lote,
        callbacks=[detencion_temprana, guardar_mejor, ajustar_lr],
        verbose=2
    )

    # Probamos con datos nunca vistos.
    error_test, precision_test = modelo.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"\nResultado final en Prueba -> Precisión: {precision_test:.4f}")

    # Guardamos el modelo completo.
    ruta_final = os.path.join(
        directorio_modelo,
        nombre_del_modelo + ".keras"
    )
    modelo.save(ruta_final)

    # Guardamos también el escalador.
    # Esto es MUY importante porque al predecir
    # debemos normalizar nuevos datos igual que en entrenamiento.
    ruta_escalador = os.path.join(directorio_modelo, "escalador.save")
    joblib.dump(escalador, ruta_escalador)

    print(f"Modelo y escalador guardados en: {directorio_modelo}")


# Punto de entrada principal del programa.
if __name__ == "__main__":
    main()
# train_hands_numbers.py
# Requisitos: pip install tensorflow pandas scikit-learn numpy joblib

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib  # Librería para guardar el escalador

# --- CONFIGURACIÓN INICIAL ---
directorio_csv = "dataset_manos/datos_manos.csv"   
directorio_modelo = "modelo_manos"
nombre_del_modelo = "modelo_numeros_manos"

# Semilla para que los resultados sean siempre los mismos al entrenar.
semilla_random = 42
# Cantidad de muestras que la red procesa antes de actualizar sus pesos.
tamano_lote = 32
# Número de veces que la red verá el dataset completo.
epocas = 100

def cargar_dataset(ruta_csv):
    """
    Carga el archivo CSV, separa la etiqueta de las coordenadas y las convierte a números.
    """
    # Leemos el archivo usando pandas
    df = pd.read_csv(ruta_csv)
    
    # Verificamos que la columna 'label' exista (es nuestra variable objetivo)
    if "label" not in df.columns:
        raise ValueError("El CSV debe tener una columna 'label' como primera columna.")
    
    # Extraemos las etiquetas (y) y las convertimos a números enteros
    etiquetas = df["label"].astype(int).to_numpy()

    # Extraemos las coordenadas (X) eliminando la columna 'label' y usando punto flotante de 32 bits
    entradas = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    
    # Validamos que tengamos los 63 landmarks (21 puntos x 3 coordenadas X,Y,Z)
    if entradas.shape[1] != 63:
        print(f"Advertencia: se detectaron {entradas.shape[1]} coordenadas (se esperaban 63).")
    
    return entradas, etiquetas

def preparar_datos(entradas, etiquetas, tamano_prueba=0.15, tamano_val=0.15, semilla=semilla_random):
    # Mezclamos los datos para que el orden de captura no afecte el aprendizaje
    entradas, etiquetas = shuffle(entradas, etiquetas, random_state=semilla)
    
    # Primera división: separamos el conjunto de Entrenamiento del resto (Temporal)
    (X_entrenamiento, X_temporal, 
     y_entrenamiento, y_temporal) = train_test_split(
        entradas, etiquetas, 
        test_size=(tamano_prueba + tamano_val), 
        random_state=semilla, 
        stratify=etiquetas # Mantiene la misma proporción de dedos/números en cada grupo
    )

    # Calculamos cuánto del grupo temporal corresponde a validación
    porcion_val = tamano_val / (tamano_prueba + tamano_val)
    
    # Segunda división: del grupo Temporal sacamos Validación y Prueba
    (X_validacion, X_prueba, 
     y_validacion, y_prueba) = train_test_split(
        X_temporal, y_temporal, 
        test_size=(1.0 - porcion_val), 
        random_state=semilla, 
        stratify=y_temporal
    )

    # Escalamos los datos para que tengan media 0 y desviación 1 (ayuda a la red a converger)
    escalador = StandardScaler()
    X_entrenamiento_esc = escalador.fit_transform(X_entrenamiento)
    X_validacion_esc = escalador.transform(X_validacion)
    X_prueba_esc = escalador.transform(X_prueba)

    return (X_entrenamiento_esc, y_entrenamiento), (X_validacion_esc, y_validacion), (X_prueba_esc, y_prueba), escalador

def construccion_modelo_neuronal_denso(dimension_entrada, num_clases=6):
    """
    Crea la arquitectura de la red neuronal profunda.
    """
    modelo = models.Sequential([
        # Capa de entrada con el tamaño de nuestros landmarks (63)
        layers.Input(shape=(dimension_entrada,)),
        
        # Primera capa densa con 256 neuronas y activación ReLU
        layers.Dense(256, activation="relu"),
        # Normalización por lotes para estabilizar el entrenamiento
        layers.BatchNormalization(),
        # Dropout apaga neuronas al azar (25%) para evitar el sobreentrenamiento (overfitting)
        layers.Dropout(0.25),
        
        # Segunda capa densa con 128 neuronas
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Tercera capa densa con 64 neuronas
        layers.Dense(64, activation="relu"),
        
        # Capa de salida con Softmax para darnos probabilidades de cada número (0 al 5)
        layers.Dense(num_clases, activation="softmax")
    ])

    # Compilamos con optimizador Adam y pérdida de entropía cruzada para clasificación
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return modelo

def main():
    # Aseguramos que los experimentos sean reproducibles
    tf.random.set_seed(semilla_random)
    np.random.seed(semilla_random)

    # Comprobamos si el archivo de datos existe
    if not os.path.exists(directorio_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {directorio_csv}")

    print("Cargando y procesando datos...")
    X, y = cargar_dataset(directorio_csv)
    clases_detectadas = len(np.unique(y))

    # Preparamos los conjuntos de datos
    (X_entreno, y_entreno), (X_val, y_val), (X_test, y_test), escalador = preparar_datos(X, y)
    
    # Construimos la red
    modelo = construccion_modelo_neuronal_denso(X_entreno.shape[1], num_clases=clases_detectadas)
    modelo.summary() # Muestra la arquitectura en consola

    # --- CALLBACKS (Funciones especiales durante entrenamiento) ---
    os.makedirs(directorio_modelo, exist_ok=True)
    ruta_mejor_modelo = os.path.join(directorio_modelo, "mejor_modelo.h5")
    
    # Detiene el entrenamiento si el error deja de bajar para no perder tiempo
    detencion_temprana = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    # Guarda automáticamente el mejor modelo basado en la validación
    guardar_mejor = callbacks.ModelCheckpoint(ruta_mejor_modelo, monitor="val_loss", save_best_only=True)
    # Baja la tasa de aprendizaje si el modelo se estanca para afinar los pesos
    ajustar_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    # --- PROCESO DE ENTRENAMIENTO ---
    historia = modelo.fit(
        X_entreno, y_entreno,
        validation_data=(X_val, y_val),
        epochs=epocas,
        batch_size=tamano_lote,
        callbacks=[detencion_temprana, guardar_mejor, ajustar_lr],
        verbose=2 # Muestra el progreso línea por línea
    )

    # Evaluación final con datos que la red nunca ha visto (Prueba)
    error_test, precision_test = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"\nResultado final en Prueba -> Precisión: {precision_test:.4f}")

    # Guardar el modelo final y el escalador
    ruta_final = os.path.join(directorio_modelo, nombre_del_modelo + ".keras")
    modelo.save(ruta_final)
    
    ruta_escalador = os.path.join(directorio_modelo, "escalador.save")
    joblib.dump(escalador, ruta_escalador)
    print(f"Modelo y escalador guardados en: {directorio_modelo}")

if __name__ == "__main__":
    main()
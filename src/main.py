#Proporciona soporte para arrays multidimensionales y matrices, así como una amplia colección de funciones matemáticas de alto nivel para operar en estos arrays.
import numpy as np
#Proporciona una infraestructura flexible y eficiente para definir y entrenar redes neuronales, así como para realizar operaciones matemáticas de alto rendimiento.
import tensorflow as tf
#El módulo pyplot proporciona una interfaz para crear gráficos de forma sencilla y rápida, imitando la funcionalidad de MATLAB.
import matplotlib.pyplot as plt

# Función a modelar: por ejemplo, y = sin(x)
def generate_data(n_samples):
    #Esto establece la semilla del generador de números aleatorios de NumPy en 0.
    # Esto asegura que cada vez que se ejecute la función, se generarán los mismos números aleatorios, lo que hace que los resultados sean reproducibles.
    np.random.seed(0)
    #valores de distribuidos uniformemente en el rango
    #Esto significa que tomará valores aleatorios entre -2pi y 2pi, lo que cubre un rango amplio de la función seno.
    x = np.random.uniform(-2*np.pi, 2*np.pi, n_samples)
    y = np.sin(x)
    return x, y

# Crear datos de entrenamiento
n_samples = 1000
x_train, y_train = generate_data(n_samples)
print(x_train,y_train)

# Crear modelo de red neuronal
'''
Valores posibles para activation
Rectified Linear Unit (ReLU, 'relu'):
Es una de las funciones de activación más utilizadas debido a su simplicidad y buen rendimiento. Introduce no linealidad en la red permitiendo el aprendizaje de relaciones más complejas.

Función de Activación Sigmoide ('sigmoid'):
Aplana los valores de entrada entre 0 y 1. Es útil en la capa de salida de un modelo de clasificación binaria, donde se desea una probabilidad de pertenencia a la clase positiva.

Función Tangente Hiperbólica ('tanh'):
Similar a la función sigmoide, pero con un rango entre -1 y 1. Es útil en las capas intermedias de una red neuronal.

Función de Activación Lineal ('linear'):
No aplica ninguna transformación a los datos de entrada. Se utiliza comúnmente en la capa de salida de modelos de regresión, donde se espera un valor numérico continuo.

Función de Activación Softmax ('softmax'):
Calcula la probabilidad de pertenencia a cada clase en un problema de clasificación multiclase. La suma de todas las salidas es igual a 1.

Función de Activación Exponencial Lineal Unit (ELU, 'elu'):
Similar a ReLU pero permite valores negativos suaves, lo que puede ayudar a reducir los problemas de muerte de neuronas.
'''
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,  input_shape=[1]),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilar el modelo
'''
Valore posibles para optimizer
Adam (tf.keras.optimizers.Adam):
Es un optimizador basado en el algoritmo de descenso de gradiente estocástico (SGD) con tasas de aprendizaje adaptativas por parámetro.
Se adapta automáticamente durante el entrenamiento, lo que lo hace adecuado para una amplia gama de problemas.
Usualmente, se usa con una tasa de aprendizaje especificada. Por ejemplo:
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

Gradiente Descendente Estocástico (Stochastic Gradient Descent, SGD) (tf.keras.optimizers.SGD):
Es el optimizador más básico. Actualiza los parámetros en la dirección opuesta al gradiente de la función de pérdida.
Se utiliza comúnmente para entrenar redes neuronales, aunque puede ser menos eficiente que otros optimizadores.
También se puede especificar una tasa de aprendizaje. Por ejemplo:
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)

Adagrad (tf.keras.optimizers.Adagrad):
Adapta la tasa de aprendizaje de cada parámetro en función de la magnitud de las actualizaciones pasadas para ese parámetro.
Es eficaz para problemas donde las características varían en diferentes escalas.
También puede tomar una tasa de aprendizaje. Por ejemplo:
optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01)

RMSprop (tf.keras.optimizers.RMSprop):
Es una versión modificada de Adagrad que normaliza la tasa de aprendizaje por la magnitud del gradiente acumulado.
Suele ser más eficaz que Adagrad en redes neuronales profundas.
Se puede especificar una tasa de aprendizaje y un factor de decaimiento. Por ejemplo:
optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-6)

Adadelta (tf.keras.optimizers.Adadelta):
Es una versión más avanzada de Adagrad que adapta la tasa de aprendizaje utilizando una ventana móvil de gradientes pasados.
No necesita especificar una tasa de aprendizaje.
Tiene menos hiperparámetros que ajustar en comparación con otros optimizadores.
optimizer=tf.keras.optimizers.Adadelta()
'''
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Mostrar curvas de entrenamiento
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Generar datos de prueba
x_test = np.linspace(-2*np.pi, 2*np.pi, 100)
y_test = np.sin(x_test)

# Predecir con el modelo entrenado
y_pred = model.predict(x_test)

# Mostrar resultados
plt.plot(x_test, y_test, label='True Function')
plt.plot(x_test, y_pred, label='Predicted Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

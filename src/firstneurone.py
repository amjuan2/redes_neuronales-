#Proporciona una infraestructura flexible y eficiente para definir y entrenar redes neuronales, así como para realizar operaciones matemáticas de alto rendimiento.
import tensorflow as tf
#Proporciona soporte para arrays multidimensionales y matrices, así como una amplia colección de funciones matemáticas de alto nivel para operar en estos arrays.
import numpy as np
#El módulo pyplot proporciona una interfaz para crear gráficos de forma sencilla y rápida, imitando la funcionalidad de MATLAB.
import matplotlib.pyplot as plt

#ejemplos de entradas y salidas
#los datos de entrada deben coincidir con los resultados para que de esta
# manera se pueda hacer match entra cada uno
celsius = np.array([-40,-10,0,8,15,22,38],dtype=float) #resultados
fahrenheit = np.array([-40,14,32,46,59,72,100],dtype=float) #respuestas

#se cran las capas dense es para que sepa que se va aconectar con el numero
# de neuronas de la capa siguiente en este caso 1 units input_shape 1 entrada
capa=tf.keras.layers.Dense(units=1,input_shape=[1])
#modelo secuencial para redes simples
# se puedes anidar modelos secuenciales para construir modelos más complejos.
#Un ejemplo práctico donde se necesiten anidar modelos es cuando deseas crear
# un modelo compuesto por submodelos que tienen propósitos diferentes. Por ejemplo,
# en un modelo de visión por computadora, puedes tener un submodelo para extraer
# características de una imagen y otro submodelo para clasificar esas características extraídas.
modelo= tf.keras.Sequential([capa])

#prepara modelo para ser entrenado, se usa el algoritmo adam para que mejore(optimizer) permite ajustar pesos y sesgos para que aprendan
#si se pasan valores muy pequeños a adam va a aprender mas lento pero valores altos pueden hacer que se pasen de la respuesta
#se usa la metrica mse mean_squared_error considera que una pequeña cantidad de errores grandes son mejor que muchos errores pequeños
'''
Valores posibles para el valor loss
**Error Cuadrático Medio (Mean Squared Error, 'mse' o 'mean_squared_error'):**
Esta es la métrica de pérdida más común para problemas de regresión. Calcula la media de los cuadrados de las diferencias entre las predicciones del modelo y los valores reales. Esta métrica es adecuada cuando se espera que la salida del modelo sea un valor numérico continuo.

Error Absoluto Medio (Mean Absolute Error, 'mae' o 'mean_absolute_error'):
Esta métrica calcula la media de las diferencias absolutas entre las predicciones del modelo y los valores reales. Es menos sensible a valores atípicos que el MSE y puede proporcionar una mejor comprensión de la magnitud de los errores.

Error Porcentual Absoluto Medio (Mean Absolute Percentage Error, 'mape' o 'mean_absolute_percentage_error'):
Calcula el error porcentual absoluto medio entre las predicciones y los valores reales. Es útil cuando se necesita comprender el error en términos porcentuales, como en problemas de pronóstico.

Error Porcentual Absoluto Medio Escalado (Symmetric Mean Absolute Percentage Error, 'smape' o 'symmetric_mean_absolute_percentage_error'):
Similar al MAPE, pero tiene en cuenta la magnitud de los valores. Es útil para comparar el error de modelos en diferentes escalas.

Error Logarítmico Cuadrático Medio (Mean Squared Logarithmic Error, 'msle' o 'mean_squared_logarithmic_error'):
Esta métrica es útil cuando las predicciones y los valores reales están en una escala logarítmica. Es menos sensible a errores en predicciones cercanas a cero que el MSE.

Entropía Cruzada Categórica (Categorical Crossentropy, 'categorical_crossentropy'):
Esta métrica se utiliza comúnmente en problemas de clasificación multiclase. Calcula la pérdida entre las distribuciones de probabilidad predichas y las distribuciones de probabilidad reales de las clases.

Entropía Cruzada Binaria (Binary Crossentropy, 'binary_crossentropy'):
Similar a la entropía cruzada categórica, pero se utiliza en problemas de clasificación binaria.
'''
'''
Adam:
Aplicaciones Generales: Adam es un optimizador muy versátil y funciona bien en una amplia 
gama de problemas de aprendizaje profundo. Si no estás seguro de qué optimizador elegir, 
Adam es una buena opción para comenzar.
Redes Neuronales Convolucionales (CNN): Adam es adecuado para entrenar CNNs en problemas 
de visión por computadora, como clasificación de imágenes.
Redes Neuronales Recurrentes (RNN): Funciona bien para entrenar RNNs en tareas de 
procesamiento de lenguaje natural (NLP), como generación de texto o traducción automática.

Stochastic Gradient Descent (SGD):
Problemas con Conjuntos de Datos Pequeños: En conjuntos de datos pequeños, 
SGD puede ser más efectivo que Adam ya que tiene menos fluctuaciones y ruido 
en la dirección del gradiente.
Problemas de Optimización Personalizados: Si necesitas una implementación de 
optimización personalizada o un control fino sobre la tasa de aprendizaje y el momentum, 
puedes utilizar SGD.

RMSprop:
Problemas con Gradientes Escasos: RMSprop es útil en problemas donde los gradientes son 
muy variables o escasos, como en el entrenamiento de modelos de lenguaje basados en palabras.
Redes Neuronales Recurrentes (RNN): Funciona bien para entrenar RNNs en problemas de secuencia, 
ya que puede adaptarse a gradientes cambiantes.

Adagrad:
Problemas con Características Escaladas de Manera Diferente: Adagrad es útil cuando las 
características de entrada varían en escalas muy diferentes.
Redes Neuronales con Sparse Data: Es adecuado para problemas donde los datos son escasos y 
los gradientes son muy variables.

Adadelta:
Problemas con Tasa de Aprendizaje Adaptable: Adadelta es útil cuando deseas un método 
de optimización que no requiere ajuste manual de la tasa de aprendizaje.
Entrenamiento de Redes Neuronales Profundas (DNN): Funciona bien en problemas de 
aprendizaje profundo con redes neuronales profundas.

Nadam:
Problemas con Grandes Conjuntos de Datos: Nadam es una buena opción para problemas 
con grandes conjuntos de datos, ya que combina las ventajas de Adam y RMSprop.
Redes Neuronales con Parametrización Sensible a la Escala: Funciona bien en problemas 
donde la escala de los parámetros puede afectar significativamente el rendimiento del modelo.

Adamax:
Redes Neuronales con Grandes Dimensiones: Adamax es adecuado para modelos con un gran 
número de parámetros, como en el entrenamiento de redes neuronales profundas con muchas capas.
Problemas de Aprendizaje a Escala: Es útil en problemas donde la escalabilidad es importante
 y se necesita una convergencia rápida.
'''
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
#comenzamos a entrenar
# se ponen las entradas(celsius), los resultados(fahrenheit) y las vueltas a dar(epochs) el verbose sirve para que no imprima tanto
historial= modelo.fit(celsius,fahrenheit,epochs=800,verbose=False)
print("modelo entrenado")


#insertar datos reales
entrada_prediccion = np.array([100, 0])
resultado = modelo.predict(entrada_prediccion.reshape(-1, 1))  # Aseguramos que la entrada sea un array de dos dimensiones
print("El resultado es: " + str(resultado) + " Fahrenheit")

print("variables")
print(capa.get_weights())


#graficar resultados funcion perdida uqe tanmal esta por vuelta
plt.xlabel("# vuelta")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.legend()
plt.show()
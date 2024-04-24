import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#ejemplos de entradas y salidas
celsius = np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100],dtype=float)

#se cran las capas dense es para que sepa que se va aconectar con el numero de neuronas de la capa siguiente en este caso 1 units input_shape 1 entrada
capa=tf.keras.layers.Dense(units=1,input_shape=[1])
#modelo secuencial para redes simples
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
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
#comenzamos a entrenar
# se ponen las entradas(celsius), los resultados(fahrenheit) y las vueltas a dar(epochs) el verbose sirve para que no imprima tanto
historial= modelo.fit(celsius,fahrenheit,epochs=800,verbose=False)
print("modelo entrenado")


#insertar datos reales
print("predict")
entrada_prediccion = np.array([100, 0])
resultado = modelo.predict(entrada_prediccion.reshape(-1, 1))  # Aseguramos que la entrada sea un array de dos dimensiones
print("El resultado es: " + str(resultado) + " Fahrenheit")

print("variables")
print(capa.get_weights())


#graficar resultados funcion perdida uqe tanmal esta por vuelta
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.legend()
plt.show()
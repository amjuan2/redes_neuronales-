import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#ejemplos de entradas y salidas
celsius = np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100],dtype=float)

#se cran las capas dense es para que sepa que se va a conectar con el numero de neuronas de
# la capa siguiente en este caso 1 units input_shape 1 entrada
oculta1 = tf.keras.layers.Dense(units = 3, input_shape = [1])
oculta2 = tf.keras.layers.Dense(units=3)
oculta3 = tf.keras.layers.Dense(units=3)
oculta4 = tf.keras.layers.Dense(units=3)
oculta5 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)

#modelo secuencial para redes simples
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

#modelo= tf.keras.Sequential([oculta1,oculta2,oculta3,oculta4,oculta5,salida])

#prepara modelo para ser entrenado
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#comenzamos a entrenar
historial= modelo.fit(celsius,fahrenheit,epochs=150,verbose=False)
print("modelo entrenado")


#insertar datos reales
print("predict")
entrada_prediccion = np.array([100.0])
resultado = modelo.predict(entrada_prediccion.reshape(-1, 1))
print("El resultado es: " + str(resultado) + " Fahrenheit")

print("variables")
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())

#graficar resultados funcion perdida uqe tanmal esta por vuelta
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.legend()
plt.show()
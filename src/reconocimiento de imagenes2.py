import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.reconocimientoimagenes import model

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalizar los valores de píxeles al rango [0, 1]
#Normalizar los píxeles ayuda a evitar problemas de divergencia o convergencia lenta durante el entrenamiento,
# especialmente en modelos que utilizan gradientes para ajustar los pesos.
train_images, test_images = train_images / 255.0, test_images / 255.0

# Seleccionar una imagen de prueba
test_index = 9
img = test_images[test_index]
true_label = test_labels[test_index]

# Hacer la predicción
#rega una dimensión extra al tensor de la imagen para que tenga la forma (1, 28, 28) y
# pueda ser procesado por el modelo
prediction = np.argmax(model.predict(np.expand_dims(img, axis=0)))

# Mostrar la imagen y la predicción
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f'Predicted: {prediction}, True: {true_label}')
plt.axis('off')
plt.show()
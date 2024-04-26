import tensorflow as tf
# Importa las capas y los modelos de Keras, que es la API de alto nivel para construir y entrenar modelos en TensorFlow.
from tensorflow.keras import layers, models
#Importa el conjunto de datos MNIST
from tensorflow.keras.datasets import mnist

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar los valores de píxeles al rango [0, 1]
#Normalizar los píxeles ayuda a evitar problemas de divergencia o convergencia lenta durante el entrenamiento,
# especialmente en modelos que utilizan gradientes para ajustar los pesos.
train_images, test_images = train_images / 255.0, test_images / 255.0

# Agregar una dimensión extra para el canal de color (escala de grises)
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# Crear el modelo de la red neuronal convolucional
# Crea un modelo secuencial, donde las capas se apilan una encima de la otra en secuencia.
model = models.Sequential([
    #Capa de convolución 2D, que aplica filtros convolucionales a las imágenes.
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    #Capa de convolución 2D, que aplica filtros convolucionales a las imágenes.
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Capa de aplanado, que convierte los mapas de características 2D en un vector
    # 1D para que puedan ser alimentados a capas densas.
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 nodos de salida para los 10 dígitos
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy en el conjunto de prueba: {test_acc}')
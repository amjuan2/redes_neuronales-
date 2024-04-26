import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Generar datos de entrenamiento

# Define la temperatura inicial de la placa en función de las coordenadas
def initial_temperature(x, y):
    return np.sin(np.pi * x / 10) * np.cos(np.pi * y / 10)

#Define la ecuación de calor, donde la temperatura en un punto (x, y) y tiempo t depende de la temperatura inicial y del tiempo.
def heat_equation(x, y, t):
    return initial_temperature(x, y) * np.exp(-t)

# Generar puntos de la placa
n_points = 50
#Crea un array de n_points puntos espaciados uniformemente entre 0 y 10.
x = np.linspace(0, 10, n_points)
y = np.linspace(0, 10, n_points)
#Crea matrices de coordenadas a partir de los arrays
X, Y = np.meshgrid(x, y)

# Visualizar la temperatura inicial
fig_init = plt.figure(figsize=(10, 8))
#Crea una figura y un subplot 3D.
ax_init = fig_init.add_subplot(111, projection='3d')
initial_temp = initial_temperature(X, Y)
#Grafica la superficie tridimensional de la temperatura inicial
ax_init.plot_surface(X, Y, initial_temp, cmap='viridis')
ax_init.set_xlabel('X')
ax_init.set_ylabel('Y')
ax_init.set_zlabel('Temperatura')
ax_init.set_title('Temperatura Inicial en una Placa')
plt.show()

# Crear conjunto de datos de entrenamiento
X_train = np.column_stack((X.ravel(), Y.ravel()))
timesteps = np.linspace(0, 1, 10)  # 10 pasos de tiempo
y_train = np.array([heat_equation(X, Y, t) for t in timesteps])

# Separar los datos de entrenamiento y prueba manualmente
train_size = int(0.8 * len(y_train))
X_train = X_train[:train_size]
y_train = y_train[:train_size]
X_test = X_train[train_size:]

# Crear y entrenar el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(n_points**2),  # Mantener la capa de salida igual
    tf.keras.layers.Reshape((n_points, n_points))  # Reshape para obtener (n_points, n_points) tensor
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Predecir la temperatura para el conjunto de prueba
try:
    y_pred = model.predict(X_test)
except Exception as e:
    print("Error occurred during prediction:", e)
    print("X_test shape:", X_test.shape)
    print("Model input shape:", model.input_shape)
    raise e

# Visualizar X_test y las predicciones para debug
print("X_test sample:", X_test[:5])
print("Predictions sample:", y_pred[:5])

# Graficar soluciones
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(timesteps)):
    ax.plot_surface(X, Y, y_pred[i], cmap='viridis', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperatura')
ax.set_title('Transferencia de Calor en una Placa')
plt.show()
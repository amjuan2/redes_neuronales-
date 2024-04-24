import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# Generar datos de entrenamiento
def initial_temperature(x, y):
    return np.sin(np.pi * x / 10) * np.cos(np.pi * y / 10)

def heat_equation(x, y, t):
    return initial_temperature(x, y) * np.exp(-t)

# Generar puntos de la placa
n_points = 50
x = np.linspace(0, 10, n_points)
y = np.linspace(0, 10, n_points)
X, Y = np.meshgrid(x, y)

# Crear conjunto de datos de entrenamiento
X_train = np.column_stack((X.ravel(), Y.ravel()))
timesteps = np.linspace(0, 1, 10)  # 10 pasos de tiempo
y_train = np.array([heat_equation(X, Y, t) for t in timesteps])
print(X_train,y_train)
# Separar los datos de entrenamiento y prueba manualmente
train_size = int(0.8 * len(y_train))
X_train, X_test = X_train[:train_size], X_train[train_size:]
y_train, y_test = y_train[:train_size], y_train[train_size:]

# Crear y entrenar el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(n_points**2)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Predecir la temperatura para el conjunto de prueba
y_pred = model.predict(X_test)

# Graficar soluciones
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(timesteps)):
    ax.plot_surface(X, Y, y_pred[i].reshape(n_points, n_points), cmap='viridis', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperatura')
ax.set_title('Transferencia de Calor en una Placa')

plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # binary: cat vs dog
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dummy training to ensure model has weights
# Replace with actual dataset
import numpy as np
X_dummy = np.random.rand(10, 256, 256, 3)
y_dummy = np.random.randint(0, 2, 10)

model.fit(X_dummy, y_dummy, epochs=1)

# Save model
model.save("model/cat_dog_model.h5")

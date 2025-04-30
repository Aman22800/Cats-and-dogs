import tensorflow as tf

# Assuming you have already trained the model
# Replace this with the actual model if necessary
model = tf.keras.models.Sequential([
    # Your model layers here
])

# Save the trained model
model.save("model/cat_dog_model.h5")
print("Model saved!")

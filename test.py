import tensorflow as tf

model = tf.keras.models.load_model("models/fifty.h5")
model.save("models/fifty.keras")
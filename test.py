import tensorflow as tf

number = ["fifty", "hundred", "hund_fif"]

for x in number:
    model = tf.keras.models.load_model(f"models/{x}.h5")
    model.save(f"models/{x}.keras")
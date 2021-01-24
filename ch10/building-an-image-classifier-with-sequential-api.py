import tensorflow as tf
import numpy as np
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255.0

print(X_train.shape, y_train.shape)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# print(model.summary())
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# flatten (Flatten)            (None, 784)               0
# _________________________________________________________________
# dense (Dense)                (None, 300)               235500
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               30100
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1010
# =================================================================
# Total params: 266,610
# Trainable params: 266,610
# Non-trainable params: 0
# _________________________________________________________________
# None

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs")

history = model.fit(X_train, y_train,
                    epochs=1,
                    batch_size=256,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_callback])

model.evaluate(X_test, y_test)

# predict proba
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba = y_proba.round(2)
# print(y_proba)

# predict class
y_pred = np.argmax(model.predict(X_new), axis=-1)
print(np.array(class_names)[y_pred])

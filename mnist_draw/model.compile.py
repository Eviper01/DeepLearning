import tensorflow as tf
import numpy as np

#Fetches all the required data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #floats
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)



classifier = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation="softmax")
])

drawer = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu",input_shape=(10,)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(784, activation="sigmoid"),
    tf.keras.layers.Reshape((28,28),input_shape=(784,))
])



# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn1 = tf.keras.losses.CategoricalCrossentropy()
loss_fn2 = tf.keras.losses.MeanSquaredError()


classifier.compile(optimizer='adam',loss=loss_fn1,metrics=['accuracy'])
classifier.fit(x_train, y_train, epochs=500,batch_size=10000)
classifier.evaluate(x_test,  y_test, verbose=2)
classifier.save("classifier.h5")
# classifier = tf.keras.models.load_model("classifier.h5")

drawer_x_train = classifier.predict(x_train)
drawer_x_test = classifier.predict(x_test)

drawer.compile(optimizer='adam',loss=loss_fn2,metrics=['accuracy'])
drawer.fit(drawer_x_train,x_train,epochs=500,batch_size=10000)
drawer.evaluate(drawer_x_test,x_test,verbose=2)
drawer.save("drawer.h5")

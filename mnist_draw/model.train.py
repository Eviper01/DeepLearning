import tensorflow as tf
import numpy as np

#Fetches all the required data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #floats
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

#tensorbaord stuff




# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


classifier = tf.keras.models.load_model("classifier.h5")
drawer = tf.keras.models.load_model("drawer.h5")


classifier.fit(x_train, y_train, epochs=500,batch_size=10000)
classifier.evaluate(x_test,  y_test, verbose=2)
classifier.save("classifier.h5")



drawer_x_train = classifier.predict(x_train)
drawer_x_test = classifier.predict(x_test)


drawer.fit(drawer_x_train,x_train,epochs=500,batch_size=10000)
drawer.evaluate(drawer_x_test,x_test,verbose=2)
drawer.save("drawer.h5")

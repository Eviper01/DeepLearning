import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': '{: 0.3f}'.format},linewidth=10000)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #floats
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

classifier = tf.keras.models.load_model("classifier.h5")
drawer = tf.keras.models.load_model("drawer.h5")

drawer_x_train = classifier.predict(x_train)
drawer_x_test = classifier.predict(x_test)

i=10
prediction = drawer.predict(np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]))
plt.imshow(prediction[0],cmap="gray_r")
plt.show()
# plt.imshow(x_test[i],cmap="gray_r")
# plt.show()

import tensorflow as tf
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format},linewidth=10000)
import time
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = tf.keras.models.load_model("out.h5")
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
start = time.time()
model.fit(x_train, y_train, epochs=100,batch_size=10000)
print("Took:",time.time()-start)
model.evaluate(x_test,  y_test, verbose=2)

model.save("out.h5")

import tensorflow as tf
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format},linewidth=10000)
model = tf.keras.models.load_model("out.h5")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
x_train =np.concatenate((x_train, np.load("dreams.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII') / 255.0))
y_train = np.concatenate((y_train, np.zeros([60000,10])+0.1))
print(x_train.shape,y_train.shape)
model.evaluate(x_test,  y_test, verbose=2)
model.fit(x_train, y_train, epochs=500,batch_size=10000)
model.evaluate(x_test,  y_test, verbose=2)
model.save("out.h5")

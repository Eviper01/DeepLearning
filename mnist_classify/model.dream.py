import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
model = tf.keras.models.load_model("out.h5")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format},linewidth=10000)

def deepdream(v):
    for index,layer in enumerate(reversed(model.get_weights())):
        if index%2 == 0:
            v = v - layer
        else:
            v = np.linalg.lstsq(layer.T,v,rcond=None)[0]

    # v = v - model.get_weights()[3] #reversing biases on output stage
    # v = np.linalg.lstsq(model.get_weights()[2].T,v,rcond=None)[0] #reversing map
    # v = v - model.get_weights()[1] # undoes bias on hidden stage
    # v = np.linalg.lstsq(model.get_weights()[0].T,v,rcond=None)[0] #reversing map
    return v.reshape(28,28)



xdreams = []
ideas = model.predict(x_train)
i = 0
for idea in ideas:
    i += 1
    if i%100==0:
        print(i)
    dream = deepdream(idea[0])
    xdreams.append((dream*255).astype(int))
xdreams = (np.stack(xdreams))
print(xdreams.shape)
np.save("dreams.npy", xdreams, allow_pickle=True, fix_imports=True)
print("done")

# print(np.load("dreams.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'))
# i=10
# dream = deepdream(model.predict(np.array([x_train[i]]))[0])
# dream = deepdream(np.array([1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]))
#
# plt.imshow(dream,cmap="gray_r")
# plt.show()
# plt.imshow(x_test[i],cmap="gray_r")
# plt.show()
# print("Normal Prediction:",model.predict(np.array([x_test[i]])))
# print("Dream Predictoin:",model.predict(np.array([dream])))

# print("\n\n\n")
# print((x_train[i]*255).astype(int))
# print("\n\n")
# print((relu_mat(dream)*255).astype(int))

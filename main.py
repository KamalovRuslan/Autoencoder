import autoencoder
import layers
import activations as afuns
import scipy.io
from sklearn.model_selection import train_test_split
import utils
import sys

shape1 = (784, 512)
shape2 = (512, 128)
shape3 = (128, 16)
shape4 = (16, 2)
shape5 = (2, 16)
shape6 = (16, 128)
shape7 = (128, 512)
shape8 = (512, 784)

l1 = layers.FCLayer(shape1, afuns.ReluActivationFunction(), use_bias=True)
l2 = layers.FCLayer(shape2, afuns.ReluActivationFunction(), use_bias=True)
l3 = layers.FCLayer(shape3, afuns.SigmoidActivationFunction(), use_bias=True)
l4 = layers.FCLayer(shape4, afuns.LinearActivationFunction(), use_bias=True)
l5 = layers.FCLayer(shape5, afuns.LinearActivationFunction(), use_bias=True)
l6 = layers.FCLayer(shape6, afuns.SigmoidActivationFunction(), use_bias=True)
l7 = layers.FCLayer(shape7, afuns.ReluActivationFunction(), use_bias=True)
l8 = layers.FCLayer(shape8, afuns.ReluActivationFunction(), use_bias=True)

mnist = scipy.io.loadmat('./data/mnist-original.mat')
data = mnist['data'].T / 255
labels = mnist['label'].T
data = data[:20000, :]
labels = labels[:20000, :]
train, test, label_train, label_test = train_test_split(data, labels, test_size=0.33)

layers = [l1, l2, l3, l4, l5, l6, l7, l8]
autoenc = autoencoder.Autoencoder(layers)

initial_weights = autoenc.net.get_weights()

# SGD
autoenc.net.set_weights(initial_weights)
hist_sgd = autoenc.run_sgd(train.T, num_epoch=5, display=True)
sys.exit(0)

# RMSprop
autoenc.net.set_weights(initial_weights)
hist_rmsprop = autoenc.run_rmsprop(train.T, num_epoch=3, display=True)

# ADAM
autoenc.net.set_weights(initial_weights)
hist_adam = autoenc.run_rmsprop(train.T, num_epoch=3, display=True)

# utils.plt_learn([hist_sgd, hist_rmsprop, hist_adam])
# utils.plt_representations(autoenc, test.T, label_test)

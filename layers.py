#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of layers used within neural networks
import numpy as np
from numpy.random import normal

class BaseLayer(object):

    def get_params_number(self):
        """
        :return num_params: number of parameters used in layer
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_weights(self):
        """
        :return w: current layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_weights(self, w):
        """
        Takes weights as a one-dimensional numpy vector and assign them to layer parameters in convenient shape,
        e.g. matrix shape for fully-connected layer
        :param w: layer weights as a numpy one-dimensional vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def set_direction(self, p):
        """
        Takes direction vector as a one-dimensional numpy vector and assign it to layer parameters direction vector
        in convenient shape, e.g. matrix shape for fully-connected layer
        :param p: layer parameters direction vector, numpy vector
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def forward(self, inputs):
        """
        Forward propagation for layer. Intermediate results are saved within layer parameters.
        :param inputs: input batch, numpy matrix of size num_inputs x num_objects
        :return outputs: layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def backward(self, derivs):
        """
        Backward propagation for layer. Intermediate results are saved within layer parameters.
        :param derivs: loss derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_derivs: loss derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_derivs: loss derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_forward(self, Rp_inputs):
        """
        Rp forward propagation for layer. Intermediate results are saved within layer parameters.
        :param Rp_inputs: Rp input batch, numpy matrix of size num_inputs x num_objects
        :return Rp_outputs: Rp layer activations, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def Rp_backward(self, Rp_derivs):
        """
        Rp backward propagation for layer.
        :param Rp_derivs: loss Rp derivatives w.r.t. layer outputs, numpy matrix of size num_outputs x num_objects
        :return input_Rp_derivs: loss Rp derivatives w.r.t. layer inputs, numpy matrix of size num_inputs x num_objects
        :return w_Rp_derivs: loss Rp derivatives w.r.t. layer parameters, numpy vector of length num_params
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_activations(self):
        """
        :return outputs: activations computed in forward pass, numpy matrix of size num_outputs x num_objects
        """
        raise NotImplementedError('This function must be implemented within child class!')


class FCLayer(BaseLayer):

    def __init__(self, shape, afun, use_bias=False):
        """
        :param shape: layer shape, a tuple (num_inputs, num_outputs)
        :param afun: layer activation function, instance of BaseActivationFunction
        :param use_bias: flag for using bias parameters
        """
        self.shape = shape
        self.afun = afun
        self.use_bias = use_bias

        scale = (1. / (shape[0] + shape[1]))
        size = shape if not use_bias else (shape[1], shape[0] + 1)
        self.W = normal(loc=0, scale=scale, size=size)

    def get_params_number(self):
        if self.use_bias:
            return (self.shape[0] + 1) * self.shape[1]
        else:
            return self.shape[0] * self.shape[1]

    def get_weights(self):
        return np.ravel(self.W)

    def set_weights(self, w):
        self.W = w.reshape(self.shape[1], self.shape[0] + self.use_bias)

    def set_direction(self, p):
        self.p = p.reshape(self.shape[1], self.shape[0] + self.use_bias)

    def get_activations(self):
        return self.zL

    def forward(self, inputs):
        self.z = inputs
        if self.use_bias:
            self.z = np.vstack((inputs, np.ones(inputs.shape[1])))

        self.inputs = self.z
        self.uL = self.W.dot(self.z)
        self.zL = self.afun.val(self.uL)
        return self.zL

    def backward(self, derivs):
        self.gz = derivs
        self.deriv_uL = self.afun.deriv(self.uL)
        self.gul = self.gz * self.deriv_uL
        gwl = self.gul.dot(self.z.T)
        gzl = self.W[:, :self.W.shape[1] - self.use_bias].T.dot(self.gul)
        return gzl, np.ravel(gwl)

    def Rp_forward(self, Rp_inputs):
        self.Rpz = Rp_inputs
        if self.use_bias:
            self.Rpz = np.vstack((Rp_inputs, np.zeros(Rp_inputs.shape[1])))

        self.Rpul = self.W.dot(self.Rpz) + self.p.dot(self.z)
        Rpzl = self.deriv_uL * self.Rpul
        return Rpzl

    def Rp_backward(self, Rp_derivs):
        Rpdul = Rp_derivs * self.deriv_uL + \
            self.gz * self.afun.second_deriv(self.uL) * self.Rpul
        Rpdwl = Rpdul.dot(self.z.T) + self.gul.dot(self.Rpz.T)
        Rpdzl = self.W[:, :self.W.shape[1] - self.use_bias].T.dot(Rpdul) +\
            self.p[:, :self.p.shape[1] - self.use_bias].T.dot(self.gul)
        return Rpdzl, np.ravel(Rpdwl)

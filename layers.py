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

    def init_normal_weights(self):
        scale = (1. / (self.shape[0] + self.shape[1]))
        if self.use_bias:
            size = (self.shape[1], self.shape[0] + 1)
        else:
            size = (self.shape[1], self.shape[0] + 0)
        self.W = normal(loc=0, scale=scale, size=size)

    def get_params_number(self):
        if self.use_bias:
            return (self.shape[0] + 1) * self.shape[1]
        else:
            return (self.shape[0] + 0) * self.shape[1]

    def get_weights(self):
        return np.ravel(self.W)

    def set_weights(self, w):
        if self.use_bias:
            self.W = w.reshape(self.shape[1], self.shape[0] + 1)
        else:
            self.W = w.reshape(self.shape[1], self.shape[0] + 0)

    def set_direction(self, p):
        if self.use_bias:
            self.p = p.reshape(self.shape[1], self.shape[0] + 1)
        else:
            self.p = p.reshape(self.shape[1], self.shape[0] + 0)

    def get_activations(self):
        return self.z

    def forward(self, inputs):
        self.inputs = inputs
        if self.use_bias:
            self.inputs = np.vstack((inputs, np.ones(inputs.shape[1])))
        self.u = self.W.dot(self.inputs)
        self.z = self.afun.val(self.u)
        return self.z

    def backward(self, derivs):
        self.derivs = derivs
        self.dg = self.afun.deriv(self.u)
        self.dL_u = self.derivs * self.dg
        dL_w = self.dL_u.dot(self.inputs.T)
        dL_z = self.W[:, :self.W.shape[1] - self.use_bias].T.dot(self.dL_u)
        return dL_z, np.ravel(dL_w)

    def Rp_forward(self, Rp_inputs):
        self.Rp_inputs = Rp_inputs
        if self.use_bias:
            self.Rp_inputs = np.vstack((Rp_inputs, np.zeros(Rp_inputs.shape[1])))
        self.Rp_u = self.W.dot(self.Rp_inputs) + self.p.dot(self.inputs)
        Rp_z = self.dg * self.Rp_u
        return Rp_z

    def Rp_backward(self, Rp_derivs):
        Rp_dL_u = Rp_derivs * self.dg + \
            self.derivs * self.afun.second_deriv(self.u) * self.Rp_u
        Rp_dL_w = Rp_dL_u.dot(self.inputs.T) + self.dL_u.dot(self.Rp_inputs.T)
        Rp_dL_z = self.W[:, :self.W.shape[1] - self.use_bias].T.dot(Rp_dL_u) +\
            self.p[:, :self.p.shape[1] - self.use_bias].T.dot(self.dL_u)
        return Rp_dL_z, np.ravel(Rp_dL_w)

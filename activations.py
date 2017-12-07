#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of activation functions used within neural networks
from scipy.special import expit
import numpy as np


class BaseActivationFunction(object):

    def val(self, inputs):
        """
        Calculates values of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def deriv(self, inputs):
        """
        Calculates first derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def second_deriv(self, inputs):
        """
        Calculates second derivatives of activation function for given inputs
        :param inputs: numpy array (vector or matrix)
        :return: result, numpy array of inputs size
        """
        raise NotImplementedError('This function must be implemented within child class!')

    def get_name(self):
        """
        Name of activation function
        """
        raise NotImplementedError('This function must be implemented within child class!')


class LinearActivationFunction(BaseActivationFunction):
    def val(self, inputs):
        return inputs

    def deriv(self, inputs):
        return np.ones_like(inputs)

    def second_deriv(self, inputs):
        return np.zeros_like(inputs)

    def get_name(self):
        return "LinearActivationFunction"


class SigmoidActivationFunction(BaseActivationFunction):
    def val(self, inputs):
        return expit(inputs)

    def deriv(self, inputs):
        _val = self.val(inputs)
        return _val * (1 - _val)

    def second_deriv(self, inputs):
        _val = self.deriv(inputs)
        _deriv = self.val(inputs)
        return _deriv(1 - 2 * _val)

    def get_name(self):
        return "SigmoidActivationFunction"


class ReluActivationFunction(BaseActivationFunction):
    def val(self, inputs):
        negative_indices = inputs < 0
        inputs[negative_indices] = 0
        return inputs

    def deriv(self, inputs):
        negative_indices = inputs < 0
        inputs[negative_indices] = 0
        positive_indices = inputs > 0
        inputs[positive_indices] = 1
        return inputs

    def second_deriv(self, inputs):
        return np.zeros_like(inputs)

    def get_name(self):
        return "ReluActivationFunction"

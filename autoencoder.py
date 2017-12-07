#!/usr/bin/python
# -*- coding: utf-8 -*-

# Implementation of autoencoder using general feed-forward neural network

import ffnet
from numpy import linalg as la
import numpy as np
import time


class Autoencoder:

    def __init__(self, layers):
        """
        :param layers: a list of fully-connected layers
        """
        self.net = ffnet.FFNet(layers)

        if self.net.layers[0].shape[0] != self.net.layers[-1].shape[1]:
            raise ValueError('In the given autoencoder number of inputs and outputs is different!')

        self.hist = {'train_loss': [],
                     'train_grad': [],
                     'test_loss':  [],
                     'test_grad':  [],
                     'epoch':      [],
                     'optimizer':  "",
                     'elaps_time': 0}

    def compute_loss(self, inputs):
        """
        Computes autoencoder loss value and loss gradient using given batch of data
        :param inputs: numpy matrix of size num_features x num_objects
        :return loss: loss value, a number
        :return loss_grad: loss gradient, numpy vector of length num_params
        """
        self.num_objects = inputs.shape[1]
        outputs = self.net.compute_outputs(inputs)
        residual = outputs - inputs
        loss = (residual ** 2).sum() / (2 * self.num_objects)
        loss_grad = self.net.compute_loss_grad(residual / self.num_objects)
        return loss, loss_grad

    def compute_hessvec(self, p):
        """
        Computes a product of Hessian and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Hp: a numpy vector of length num_params
        """
        self.net.set_direction(p)
        Rp_outputs = self.net.compute_Rp_outputs() / self.num_objects
        return self.net.compute_loss_Rp_grad(Rp_outputs)

    def compute_gaussnewtonvec(self, p):
        """
        Computes a product of Gauss-Newton Hessian approximation and given direction vector
        :param p: direction vector, a numpy vector of length num_params
        :return Gp: a numpy vector of length num_params
        """
        self.net.set_direction(p)
        Rp_outputs = self.net.compute_Rp_outputs() / self.num_objects
        return self.net.compute_loss_grad(Rp_outputs)

    def batch_generator(self, inputs, minibatch_size):
        l = inputs.shape[1]
        for ndx in range(0, l, minibatch_size):
            yield inputs[:, ndx:min(ndx + minibatch_size, l)]

    def init_weights(self):
        for layer in self.net.layers:
            layer.init_normal_weights()

    def summary(self):
        print("Layers number: {}".format(len(self.net.layers)))
        total_params_number = 0
        for i, layer in enumerate(self.net.layers):
            afun_name = layer.afun.get_name()
            shape = layer.shape
            params_n = layer.get_params_number()
            total_params_number += params_n
            info = "layer_{0} | activation {1:25s} | shape {2:10s} | params number {3}".format(
                    i+1,      afun_name,     str(shape),    params_n
            )
            print(info)
        print("Total params number: {}". format(total_params_number))

    def run_sgd(self, inputs, step_size=0.01, momentum=0.9, num_epoch=200,
                minibatch_size=100, l2_coef=1e-5, test_inputs=None, display=False):
        """
        Stochastic gradient descent optimization
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param momentum: momentum coefficient, number
        :param num_epoch: number of training epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs, the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each peoch, list
        """
        self.hist = {'train_loss': [],
                     'train_grad': [],
                     'test_loss':  [],
                     'test_grad':  [],
                     'epoch':      [],
                     'optimizer':  "",
                     'elaps_time': 0}

        self.summary()
        print("\nOptimizer SGD\n")
        start_time = time.time()
        w = self.net.get_weights()
        d_w = np.zeros_like(w)
        train_loss_list = []
        train_grad_list = []
        if test_inputs is not None:
            test_loss_list = []
            test_grad_list = []
        batch_num = inputs.shape[1] / minibatch_size
        for epoch in range(num_epoch):
            train_loss = 0
            train_grad = 0
            for batch in self.batch_generator(inputs, minibatch_size):
                loss, loss_grad = self.compute_loss(batch)
                loss_grad += l2_coef * w
                train_loss += loss
                train_grad += la.norm(loss_grad, 2)

                d_w = momentum * d_w + step_size * loss_grad
                w -= d_w
                self.net.set_weights(w)

            train_loss_list.append(train_loss / batch_num)
            train_grad_list.append(train_grad / batch_num)

            if test_inputs is not None:
                test_loss, test_loss_grad = self.compute_loss(test_inputs)
                test_loss_list.append(test_loss)
                test_grad_list.append(la.norm(test_loss_grad, 2))

            if display:
                _loss = train_loss_list[-1]
                _loss_grad = train_grad_list[-1]
                info = 'epoch: {0}  loss: {1:.4f} loss_grad: {2:.4f}'.format(
                        epoch,     _loss,        _loss_grad
                        )
                if test_inputs is not None:
                    _val_loss = test_loss_list[-1]
                    _val_loss_grad = test_grad_list[-1]
                    test_info = 'val_loss: {0:.4f} val_loss_grad: {1:.4f}'.format(
                                _vall_loss,       _val_loss_grad
                                )
                    info += test_info
                print(info)

        self.hist['train_loss'] = train_loss_list
        self.hist['train_grad'] = train_grad_list
        if test_inputs is not None:
            self.hist['test_loss'] = test_loss_list
            self.hist['test_grad'] = test_grad_list
        self.hist['elaps_time'] = time.time() - start_time
        self.hist['epoch'] = list(range(num_epoch))
        self.hist['optimizer'] = "SGD"
        return self.hist

    def run_rmsprop(self, inputs, step_size=0.01, num_epoch=200,
                    minibatch_size=100, l2_coef=1e-5, test_inputs=None, display=False):
        """
        RMSprop stochastic optimization
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param num_epoch: number of training epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs, the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each peoch, list
        """
        self.hist = {'train_loss': [],
                     'train_grad': [],
                     'test_loss':  [],
                     'test_grad':  [],
                     'epoch':      [],
                     'optimizer':  "",
                     'elaps_time': 0}
        self.summary()
        print("\nOptimizer RMSProp\n")
        start_time = time.time()
        gamma, epsilon = .9, 1e-9
        w = self.net.get_weights()
        d_w = np.zeros_like(w)
        train_loss_list = []
        train_grad_list = []
        if test_inputs is not None:
            test_loss_list = []
            test_grad_list = []
        batch_num = inputs.shape[1] / minibatch_size
        for epoch in range(num_epoch):
            train_loss = 0
            train_grad = 0
            for batch in self.batch_generator(inputs, minibatch_size):
                loss, loss_grad = self.compute_loss(batch)
                train_loss += loss
                loss_grad += l2_coef * w
                train_grad += la.norm(loss_grad, 2)

                d_w = gamma * d_w + (1 - gamma) * (loss_grad ** 2)
                w -= step_size * loss_grad / ((d_w + epsilon) ** .5)
                self.net.set_weights(w)

            train_loss_list.append(train_loss / batch_num)
            train_grad_list.append(train_grad / batch_num)
            if test_inputs is not None:
                test_loss, test_loss_grad = self.compute_loss(test_inputs)
                test_loss_list.append(test_loss)
                test_grad_list.append(la.norm(test_loss_grad, 2))

            if display:
                _loss = train_loss_list[-1]
                _loss_grad = train_grad_list[-1]
                info = 'epoch: {0}  loss: {1:.4f} loss_grad: {2:.4f}'.format(
                        epoch,     _loss,        _loss_grad
                        )
                if test_inputs is not None:
                    _val_loss = test_loss_list[-1]
                    _val_loss_grad = test_grad_list[-1]
                    test_info = 'val_loss: {0:.4f} val_loss_grad: {1:.4f}'.format(
                                _vall_loss,       _val_loss_grad
                                )
                    info += test_info
                print(info)

        self.hist['train_loss'] = train_loss_list
        self.hist['train_grad'] = train_grad_list
        if test_inputs is not None:
            self.hist['test_loss'] = test_loss_list
            self.hist['test_grad'] = test_grad_list
        self.hist['elaps_time'] = time.time() - start_time
        self.hist['epoch'] = list(range(num_epoch))
        self.hist['optimizer'] = "RMSProp"
        return self.hist

    def run_adam(self, inputs, step_size=0.01, num_epoch=200,
                 minibatch_size=100, l2_coef=1e-5, test_inputs=None, display=False):
        """
        ADAM stochastic optimization
        :param inputs: training sample, numpy matrix of size num_features x num_objects
        :param step_size: step size, number
        :param num_epoch: maximal number of epochs, number
        :param minibatch_size: number of objects in each minibatch, number
        :param l2_coef: L2 regularization coefficient, number
        :param test_inputs: testing sample, numpy matrix of size num_features x num_test_objects
        :param display: print information for epochs, bool
        :return results: a dictionary with results per optimization epochs, the following key, values are possible:
            'train_loss': loss values for last train batch for each epoch, list
            'train_grad': norm of loss gradients for last train batch for each epoch, list
            'test_loss': loss values for testing sample after each epoch, list
            'test_grad': norm of loss gradients for testing sample after each peoch, list
        """
        self.hist = {'train_loss': [],
                     'train_grad': [],
                     'test_loss':  [],
                     'test_grad':  [],
                     'epoch':      [],
                     'optimizer':  "",
                     'elaps_time': 0}
        self.summary()
        print("\nOptimizer Adam\n")
        start_time = time.time()
        w = self.net.get_weights()
        b1, b2, epsilon = .001, .9, 1e-8
        train_loss_list = []
        train_grad_list = []
        if test_inputs is not None:
            test_loss_list = []
            test_grad_list = []
        batch_num = inputs.shape[1] / minibatch_size
        m, v, t = 0, 0, 0
        for epoch in range(num_epoch):
            train_loss = 0
            train_grad = 0
            for batch in self.batch_generator(inputs, minibatch_size):
                t += 1
                loss, loss_grad = self.compute_loss(batch)
                train_loss += loss
                train_grad += la.norm(loss_grad, 2)
                loss_grad += l2_coef * w

                m = b1 * m + (1 - b1) * loss_grad
                v = b2 * v + (1 - b2) * (loss_grad ** 2)
                mn = m / (1 - b1 ** t)
                vn = v / (1 - b2 ** t)
                w -= step_size * mn / (vn ** .5 + epsilon)
                self.net.set_weights(w)

            train_loss_list.append(train_loss / batch_num)
            train_grad_list.append(train_grad / batch_num)
            if test_inputs is not None:
                test_loss, test_loss_grad = self.compute_loss(test_inputs)
                test_loss_list.append(test_loss)
                test_grad_list.append(la.norm(test_loss_grad, 2))

            if display:
                _loss = train_loss_list[-1]
                _loss_grad = train_grad_list[-1]
                info = 'epoch: {0}  loss: {1:.4f} loss_grad: {2:.4f}'.format(
                        epoch,     _loss,        _loss_grad
                        )
                if test_inputs is not None:
                    _val_loss = test_loss_list[-1]
                    _val_loss_grad = test_grad_list[-1]
                    test_info = 'val_loss: {0:.4f} val_loss_grad: {1:.4f}'.format(
                                _vall_loss,       _val_loss_grad
                                )
                    info += test_info
                print(info)

        self.hist['train_loss'] = train_loss_list
        self.hist['train_grad'] = train_grad_list
        if test_inputs is not None:
            self.hist['test_loss'] = test_loss_list
            self.hist['test_grad'] = test_grad_list
        self.hist['elaps_time'] = time.time() - start_time
        self.hist['epoch'] = list(range(num_epoch))
        self.hist['optimizer'] = "ADAM"
        return self.hist

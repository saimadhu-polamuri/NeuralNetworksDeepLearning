#!/usr/bin/env python
# neuralnet.py
# Author : Saimadhu
# Date: 20-March-2017
# About: Modeling Neural nets

# Required Python Packages
import numpy as np


class Network(object):
    """
    Simple Neural network for predicting the digits
    """

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        """
        To get the updated activation scores
        :param z:
        :return:
        """
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        """
        Returns the output of the network if a is input
        :param a:
        :return:
        """

        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)


test = Network([2, 3, 1])

print "Biases :: {}".format(test.biases)
print "Weights :: {}".format(test.weights)

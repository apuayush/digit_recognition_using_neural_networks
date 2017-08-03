import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        """
        This is general for setting number of neurons we want to set in a
        particular layer.
        :param sizes: contains number of neurons in respective layers
        """

        self.sizes = sizes
        self.num_of_layers = len(sizes)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """
        Returns the output of the network if 'a' is input
        mainly for learning through training data
        :param a: input
        :return: a i.e modified weights and biases
        """
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,training_data, epochs,
            mini_batch_size, eta,
            test_data=None):
        """
        This is Stochastic gradient descent form of learning
        We are taking a mini_batch out of 60,000 data sets to train our
        model through selecting only those weights and biases which make cost
        function minimum
        :param training_data: list of tuples (x,y) containing training inputs and
        their desired outputs
        :param epochs: randomly chosen numbers
        :param mini_batch_size: size of the miniature training data
        :param eta: learning rate
        :param test_data: network will test itself after each epoch using test_data
        :return:
        """

        if test_data:
            n_test = len(test_data)
        n= len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k: k+mini_batch_size] for k in xrange(
                0, n ,mini_batch_size
            )]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1}/ {2}".format(
                    j,self.evaluate(test_data), n_test
                )

            else :
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weight and biases by applying gradient descend
         using *back propagation*
        :param mini_batch - a list of tuples '(x,y)'
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

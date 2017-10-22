#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A simple neural networki using MNIST data.

.. _Original module repository:
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/
        01_Simple_Linear_Model.ipynb

"""

import numpy as np
import tensorflow as tf


class ClassifierNN:
    ACTIVATIONS = ['relu']

    def __init__(self, hidden_layers=None, input_data=None,
                 activation_func='relu'):
        self.input_data = input_data

        if hidden_layers is not None:
            self.build_layers(hidden_layers, activation_func)
        else:
            self.layers = None

    def build_layers(self, hidden_layers=None, activation_func='relu'):
        if self.input_data is None:
            raise Exception('No data to determine the dimensions of the neural'
                            ' network.')

        if activation_func not in self.ACTIVATIONS:
            raise Exception('The activation method is not valid')
        else:
            self.activation_func = activation_func

        if hidden_layers is None:
            hidden_layers = []

        self.data_shape = self.input_data.shape

        self.data_elements = np.prod(self.data_shape)

        self.input_placeholder = tf.placeholder('float', [None,
                                                          self.data_elements])

        # self.ouput_placeholder = tf.placeholder('float')

        self.layers = []

        n_last_layer = self.data_elements
        last_layer_output = self.input_placeholder

        for n in hidden_layers + [self.input_data.N_CATEGORIES]:
            self.layers.append(
                {'weights': tf.Variable(tf.random_normal([n_last_layer, n])),
                 'biases': tf.Variable(tf.random_normal([n]))})

            layer = tf.add(tf.matmul(last_layer_output,
                                     self.layers[-1]['weights']),
                           self.layers[-1]['biases'])

            if self.activation_func == 'relu':
                layer = tf.nn.relu(layer)

            self.layers[-1]['output'] = layer

            last_layer_output = layer
            n_last_layer = n

        self.ouput_placeholder = self.layers[-1]['output']

    def train_network(self, batch_size=100, epochs=10, train_func='adam',
                      verbose=True):
        if self.layers is None:
            raise Exception('The neural network was not initialized.')

        label_placeholder = tf.placeholder('float')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.ouput_placeholder, labels=label_placeholder))

        if train_func == 'adam':
            optimizer = tf.train.AdamOptimizer().minimize(cost)
        else:
            raise NotImplementedError('This train function is not '
                                      'implemented.')

        with tf.Session() as sess:
            # Starting the tf session and initializing the tf variables
            sess.run(tf.global_variables_initializer())
            result = []

            for epoch in range(epochs):
                epoch_loss = 0
                for _ in range(int(len(self.input_data) / batch_size)):
                    batch_x, batch_y = self.input_data.next_batch(batch_size)
                    # print('batch_x', batch_x)
                    # print('batch_y', batch_y)
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={self.input_placeholder: batch_x,
                                               label_placeholder: batch_y})

                    epoch_loss += c
                if verbose:
                    print('Epoch', epoch + 1, 'completed out of',
                          str(epochs) + '. Loss:', epoch_loss)

            labels = np.array(self.input_data.train_dataset['labels'])

            labels = labels.reshape((len(labels), 1))

            correct = tf.equal(tf.argmax(self.ouput_placeholder, 1),
                               tf.constant(labels))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            data = self.input_data.train_dataset['data_array']

            if verbose:
                print('Accuracy: ',
                      str(accuracy.eval({self.input_placeholder: data})))

            '''
            Concatenating the weights of the layers on the result array.

            To get the contents of the tf.Variable tensors we need to use the
            `eval` method, which will return the respective np.array.
            '''
            for layer in self.layers:
                result.append({'weights': layer['weights'].eval(session=sess),
                               'biases': layer['biases'].eval(session=sess)})

        self.weights = result

        return result

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A simple neural networki using MNIST data.

.. _Original module repository:
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/
        01_Simple_Linear_Model.ipynb

"""

import numpy as np
import tensorflow as tf


np.set_printoptions(threshold=np.nan)


class ClassifierNN:
    ACTIVATION_FUNC = ['relu', 'sigmoid']
    TRAIN_FUNC = ['adam', 'gradientdescent']

    def __init__(self, hidden_layers=None, input_data=None,
                 activation_func='relu'):
        self.input_data = input_data

        if hidden_layers is not None:
            self.build_layers(hidden_layers, activation_func)
        else:
            self.layers = None

    def build_layers(self, hidden_layers=None, activation_func='sigmoid',
                     verbose=True):
        if self.input_data is None:
            raise Exception('No data to determine the dimensions of the neural'
                            ' network.')

        if activation_func not in self.ACTIVATION_FUNC:
            raise Exception('The activation method is not valid')
        else:
            self.activation_func = activation_func

        if hidden_layers is None:
            hidden_layers = []

        self.data_shape = self.input_data.shape

        self.data_elements = np.prod(self.data_shape)

        if verbose:
            print('Building the input layer.')

        self.input_placeholder = tf.placeholder('float', [None,
                                                          self.data_elements])

        # self.ouput_placeholder = tf.placeholder('float')

        self.layers = []

        n_last_layer = self.data_elements
        last_layer_output = self.input_placeholder

        for n in hidden_layers + [self.input_data.N_CATEGORIES]:
            if verbose:
                print('Building a layer with', n, 'perceptrons.')
            self.layers.append(
                {'weights': tf.Variable(tf.random_normal([n_last_layer, n])),
                 'biases': tf.Variable(tf.random_normal([n]))})

            layer = tf.add(tf.matmul(last_layer_output,
                                     self.layers[-1]['weights']),
                           self.layers[-1]['biases'])

            if self.activation_func == 'relu':
                layer = tf.nn.relu(layer)
            elif self.activation_func == 'sigmoid':
                layer = tf.nn.sigmoid(layer)

            self.layers[-1]['output'] = layer

            last_layer_output = layer
            n_last_layer = n

        self.ouput_placeholder = self.layers[-1]['output']

        if verbose:
            print('Size of the neural network:', len(self.layers) + 1)

    def train_network(self, batch_size=100, epochs=10, train_func='adam',
                      verbose=True, cost_func='cross_entropy',
                      learning_rate=0.1):
        if self.layers is None:
            raise Exception('The neural network was not initialized.')

        label_placeholder = tf.placeholder('float')

        if cost_func == 'cross_entropy':
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.ouput_placeholder, labels=label_placeholder))
        elif cost_func == 'rmse':
            cost = tf.sqrt(tf.reduce_mean(tf.square(
                tf.subtract(self.ouput_placeholder, label_placeholder))))

        if train_func == 'adam':
            optimizer = \
                tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(cost)
        elif train_func == 'gradientdescent':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate).minimize(cost)
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

            labels = np.array(self.input_data.test_dataset['labels'])

            labels = labels.reshape((len(labels), 1))

            correct = tf.equal(tf.argmax(self.ouput_placeholder, 1),
                               tf.constant(labels))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            data = self.input_data.test_dataset['data_array']

            estimated = np.argmax(self.ouput_placeholder.eval(
                            {self.input_placeholder: data}), axis=1)
            if verbose:
                print('Accuracy: ',
                      str(accuracy.eval({self.input_placeholder: data})))
                print('Estimated:', estimated)

            '''
            Concatenating the weights of the layers on the result array.

            To get the contents of the tf.Variable tensors we need to use the
            `eval` method, which will return the respective np.array.
            '''
            for layer in self.layers:
                result.append({'weights': layer['weights'].eval(session=sess),
                               'biases': layer['biases'].eval(session=sess)})

        self.weights = result

        with open('results.txt', 'w') as f:
            f.write(str(result))
            f.write(str(estimated))
            f.write(str(labels))

        return result

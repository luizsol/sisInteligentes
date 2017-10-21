#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A simple neural networki using MNIST data.

.. _Original module repository:
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/
        01_Simple_Linear_Model.ipynb

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from cifar_data import CifarData

# Downloading and importing the MNIST data
data = CifarData()
data.load_cifar10_data()

# Displaying the downloaded data
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train_dataset['labels'])))
print("- Test-set:\t\t{}".format(len(data.test_dataset['labels'])))

# The data-set has been loaded as so-called One-Hot encoding. This means the
# labels have been converted from a single number to a vector whose length
# equals the number of possible classes. All elements of the vector are zero
# except for the $i$'th element which is one and means the class is $i$.
# For example, the One-Hot encoded labels for the first 5 images in the
# test-set are:
print(data.test_dataset['labels'][0:5])

# Dimensions

img_size = data.LOADED_IMG_HEIGHT

# Images are stored in one-dimensional arrays of this length.
img_size_flat = data.LOADED_IMG_HEIGHT * data.LOADED_IMG_WIDTH * \
    data.LOADED_IMG_DEPTH

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Helper-function for plotting images
# Function used to plot 9 images in a 3x3 grid, and writing the true and
# predicted classes below each image.


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    plt.ion()  # Activating interactive mode ploting

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])


# Usage example
# Get the first images from the test-set.
images = data.test_dataset['data'][0:9]

# Get the true classes for those images.
cls_true = data.test_dataset['labels'][0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


# Placeholder variables

# Placeholder variables serve as the input to the graph that we may change each
# time we execute the graph. We call this feeding the placeholder variables and
# it is demonstrated further below.

# First we define the placeholder variable for the input images. This allows us
# to change the images that are input to the TensorFlow graph. This is a
# so-called tensor, which just means that it is a multi-dimensional vector or
# matrix. The data-type is set to float32 and the shape is set to
# [None, img_size_flat], where None means that the tensor may hold an arbitrary
# number of images with each image being a vector of length img_size_flat.
x = tf.placeholder(tf.float32, [None, img_size_flat])

# Next we have the placeholder variable for the true labels associated with the
# images that were input in the placeholder variable x. The shape of this
# placeholder variable is [None, num_classes] which means it may hold an
# arbitrary number of labels and each label is a vector of length num_classes
# which is 10 in this case.
y_true = tf.placeholder(tf.float32, [None, num_classes])

# Finally we have the placeholder variable for the true class of each image in
# the placeholder variable x. These are integers and the dimensionality of this
# placeholder variable is set to [None] which means the placeholder variable is
# a one-dimensional vector of arbitrary length.
y_true_cls = tf.placeholder(tf.int64, [None])


# Variables to be optimized

# Apart from the placeholder variables that were defined above and which serve
# as feeding input data into the model, there are also some model variables
# that must be changed by TensorFlow so as to make the model perform better on
# the training data.

# The first variable that must be optimized is called weights and is defined
# here as a TensorFlow variable that must be initialized with zeros and whose
# shape is [img_size_flat, num_classes], so it is a 2-dimensional tensor (or
# matrix) with img_size_flat rows and num_classes columns.
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))

# The second variable that must be optimized is called biases and is defined as
# a 1-dimensional tensor (or vector) of length num_classes.
biases = tf.Variable(tf.zeros([num_classes]))


# Model

# This simple mathematical model multiplies the images in the placeholder
# variable x with the weights and then adds the biases.

# The result is a matrix of shape [num_images, num_classes] because x has shape
# [num_images, img_size_flat] and weights has shape
# [img_size_flat, num_classes], so the multiplication of those two matrices is
# a matrix with shape [num_images, num_classes] and then the biases vector is
# added to each row of that matrix.

# Note that the name logits is typical TensorFlow terminology, but other people
# may call the variable something else.

logits = tf.matmul(x, weights) + biases

# Now logits is a matrix with num_images rows and num_classes columns, where
# the element of the $i$'th row and $j$'th column is an estimate of how likely
# the $i$'th input image is to be of the $j$'th class.

# However, these estimates are a bit rough and difficult to interpret because
# the numbers may be very small or large, so we want to normalize them so that
# each row of the logits matrix sums to one, and each element is limited
# between zero and one. This is calculated using the so-called softmax function
# and the result is stored in y_pred.

y_pred = tf.nn.softmax(logits)

# The predicted class can be calculated from the y_pred matrix by taking the
# index of the largest element in each row.

y_pred_cls = tf.argmax(y_pred, dimension=1)


# Cost-function to be optimized

# To make the model better at classifying the input images, we must somehow
# change the variables for weights and biases. To do this we first need to know
# how well the model currently performs by comparing the predicted output of
# the model y_pred to the desired output y_true.

# The cross-entropy is a performance measure used in classification. The
# cross-entropy is a continuous function that is always positive and if the
# predicted output of the model exactly matches the desired output then the
# cross-entropy equals zero. The goal of optimization is therefore to minimize
# the cross-entropy so it gets as close to zero as possible by changing the
# weights and biases of the model.

# TensorFlow has a built-in function for calculating the cross-entropy. Note
# that it uses the values of the logits because it also calculates the softmax
# internally.

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_true)

# We have now calculated the cross-entropy for each of the image
# classifications so we have a measure of how well the model performs on each
# image individually. But in order to use the cross-entropy to guide the
# optimization of the model's variables we need a single scalar value, so we
# simply take the average of the cross-entropy for all the image
# classifications.

cost = tf.reduce_mean(cross_entropy)


# Optimization method

# Now that we have a cost measure that must be minimized, we can then create an
# optimizer. In this case it is the basic form of Gradient Descent where the
# step-size is set to 0.5.

# Note that optimization is not performed at this point. In fact, nothing is
# calculated at all, we just add the optimizer-object to the TensorFlow graph
# for later execution.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)


# Performance measures

# We need a few more performance measures to display the progress to the user.

# This is a vector of booleans whether the predicted class equals the true
# class of each image.

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# This calculates the classification accuracy by first type-casting the vector
# of booleans to floats, so that False becomes 0 and True becomes 1, and then
# calculating the average of these numbers.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# TensorFlow Run

# Create TensorFlow session

# Once the TensorFlow graph has been created, we have to create a TensorFlow
# session which is used to execute the graph.

session = tf.Session()

# Initialize variables

# The variables for weights and biases must be initialized before we start
# optimizing them.
session.run(tf.global_variables_initializer())

# Helper-function to perform optimization iterations

# There are 50.000 images in the training-set. It takes a long time to
# calculate the gradient of the model using all these images. We therefore use
# Stochastic Gradient Descent which only uses a small batch of images in each
# iteration of the optimizer.

batch_size = 100

# Function for performing a number of optimization iterations so as to
# gradually improve the weights and biases of the model. In each iteration,
# a new batch of data is selected from the training-set and then TensorFlow
# executes the optimizer using those training samples.


def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

# Helper-functions to show performance

# Dict with the test-set data to be used as input to the TensorFlow graph.
# Note that we must use the correct names for the placeholder variables in the
# TensorFlow graph.


feed_dict_test = {x: data.test_dataset['data'],
                  y_true: data.test_dataset['labels'],
                  y_true_cls: data.test_dataset['cls']}

# Function for printing the classification accuracy on the test-set.


def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

# Function for printing and plotting the confusion matrix using scikit-learn.


def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = data.test_dataset['cls']

    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

# Helper-function to plot the model weights

# Function for plotting the weights of the model. 10 images are plotted, one
# for each digit that the model is trained to recognize.


def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i < 10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

# Performance before any optimization

# The accuracy on the test-set is 9.8%. This is because the model has only been
# initialized and not optimized at all, so it always predicts that the image
# shows a zero digit, as demonstrated in the plot below, and it turns out that
# 9.8% of the images in the test-set happens to be zero digits.


print_accuracy()

# Performance after 1 optimization iteration

# Already after a single optimization iteration, the model has increased its
# accuracy on the test-set to 40.7% up from 9.8%. This means that it
# mis-classifies the images about 6 out of 10 times, as demonstrated on a few
# examples below.

optimize(num_iterations=1)

print_accuracy()

# The weights can also be plotted as shown below. Positive weights are red and
# negative weights are blue. These weights can be intuitively understood as
# image-filters.

# For example, the weights used to determine if an image shows a zero-digit
# have a positive reaction (red) to an image of a circle, and have a negative
# reaction (blue) to images with content in the centre of the circle.

# Similarly, the weights used to determine if an image shows a one-digit react
# positively (red) to a vertical line in the centre of the image, and react
# negatively (blue) to images with content surrounding that line.

# Note that the weights mostly look like the digits they're supposed to
# recognize. This is because only one optimization iteration has been performed
# so the weights are only trained on 100 images. After training on several
# thousand images, the weights become more difficult to interpret because they
# have to recognize many variations of how digits can be written.

plot_weights()

# Performance after 10 optimization iterations

# We have already performed 1 iteration.
optimize(num_iterations=9)

print_accuracy()

plot_weights()

# Performance after 1000 optimization iterations

# After 1000 optimization iterations, the model only mis-classifies about one
# in ten images. As demonstrated below, some of the mis-classifications are
# justified because the images are very hard to determine with certainty even
# for humans, while others are quite obvious and should have been classified
# correctly by a good model. But this simple model cannot reach much better
# performance and more complex models are therefore needed.

# We have already performed 10 iterations.
optimize(num_iterations=990)

print_accuracy()

# The model has now been trained for 1000 optimization iterations, with each
# iteration using 100 images from the training-set. Because of the great
# variety of the images, the weights have now become difficult to interpret and
# we may doubt whether the model truly understands how digits are composed from
# lines, or whether the model has just memorized many different variations of
# pixels.

plot_weights()

# We can also print and plot the so-called confusion matrix which lets us see
# more details about the mis-classifications. For example, it shows that images
# actually depicting a 5 have sometimes been mis-classified as all other
# possible digits, but mostly either 3, 6 or 8.

print_confusion_matrix()

# We are now done using TensorFlow, so we close the session to release its
# resources.

print(weights)

session.close()

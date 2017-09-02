#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import copy

import cv2
import numpy as np

class ImageProcessor(object):
    """A class responsible for storing and processing images.

    Attributes:
        image: A virtual attribute to implement the setter and getter of _image
        _image: The real attribute responsible for storing the class image
    """

    def __init__(self, image=None):
        """Inits ImageProcessor with an optional image."""
        if image:
            self.image = image
        else:
            self._image = None

    @property
    def image(self):
        """The image getter.

        The true image object is incapsulated on the _image attribute,
        therefore the user must use the image virtual attribute to interact
        with image stored in this class

        Returns:
            A numpy.ndarray containing the image.
        """
        return self._image

    @image.setter
    def image(self, value):
        """The image setter.

        Args:
            value: either a numpy.ndarray containing a image or the path to an
                image.

        Raises:
            TypeError: The value is neither an image nor the path to an image.
        """
        if isinstance(value, np.ndarray):
            self._image = value
        elif isinstance(value, str):
            self._image = cv2.imread(value, cv2.IMREAD_UNCHANGED)
        else:
            raise TypeError

    def has_image(self):
        """Determines whether an image is already stored.

        Returns:
            True if there's already an image stored, False otherwise.
        """
        if self.image is None:
            return False
        return True

    def show_image(self, window_name='image'):
        """Displays the current image.

        Keyword Args:
            window_name (='image'): the name of the window in which the image
                will be displayed
        """
        cv2.imshow(window_name, self.image)

    def save_image(self, path='image_processor_output.bmp'):
        """Saves the current image on the file system.

        Keyword Args:
            path (='image_processor_output'): the path where the file should be
                saved
        """
        cv2.imwrite(path, self.image)

    def get_laplacian(self, data_type=cv2.CV_32F):
        """Calculates the Laplacian of the image.

        Keyword Args:
            data_type (=cv2.CV_32F): the datatype to be used on the calculation

        Returns:
            The resulting Laplacian matrix.
        """
        return cv2.Laplacian(self.image, data_type)

    def apply_laplacian(self, data_type=cv2.CV_32F):
        """Substitutes the current image for it's Laplacian.

        Keyword Args:
            data_type (=cv2.CV_32F): the datatype to be used on the calculation
        """
        self.image = self.get_laplacian(data_type=data_type)

    def show_laplacian(self, window_name='image', data_type=cv2.CV_32F):
        """Shows the Laplacian of the image.

        Keyword Args:
            window_name (='image'): the name of the window in which the image
                will be displayed
            data_type (=cv2.CV_32F): the datatype to be used on the calculation
        """
        cv2.imshow(window_name, self.get_laplacian(data_type=data_type))

    def get_x_grad(self, kernel_size=5, data_type=cv2.CV_32F):
        """Calculates the Sobel gradient of the image on the X axis.

        Keyword Args:
            kernel_size (=5): the kernel size of the Sobel filter
            data_type (=cv2.CV_32F): the datatype to be used on the calculation

        Returns:
            The resulting unidirectional Gradient matrix.
        """
        return cv2.Sobel(self.image, data_type, 1, 0, kernel_size)

    def apply_x_grad(self, kernel_size=5, data_type=cv2.CV_32F):
        """Substitutes the current image for it's Sobel's X axis gradient.

        Keyword Args:
            kernel_size (=5): the kernel size of the Sobel filter
            data_type (=cv2.CV_32F): the datatype to be used on the calculation
        """
        self.image = self.get_x_grad(kernel_size=kernel_size,
                                     data_type=data_type)

    def show_x_grad(self, window_name='image', kernel_size=5,
                    data_type=cv2.CV_32F):
        """Shows the Sobel's X axis gradient of the image.

        Keyword Args:
            window_name (='image'): the name of the window in which the image
                will be displayed
            kernel_size (=5): the kernel size of the Sobel filter
            data_type (=cv2.CV_32F): the datatype to be used on the calculation
        """
        cv2.imshow(window_name, self.get_x_grad(kernel_size=kernel_size,
                                                data_type=data_type))

    def get_y_grad(self, kernel_size=5, data_type=cv2.CV_32F):
        """Calculates the Sobel gradient of the image on the Y axis.

        Keyword Args:
            kernel_size (=5): the kernel size of the Sobel filter
            data_type (=cv2.CV_32F): the datatype to be used on the calculation

        Returns:
            The resulting unidirectional Gradient matrix.
        """
        return cv2.Sobel(self.image, data_type, 0, 2, kernel_size)

    def apply_y_grad(self, kernel_size=5, data_type=cv2.CV_32F):
        """Substitutes the current image for it's Sobel's Y axis gradient.

        Keyword Args:
            kernel_size (=5): the kernel size of the Sobel filter
            data_type (=cv2.CV_32F): the datatype to be used on the calculation
        """
        self.image = self.get_y_grad(kernel_size=kernel_size,
                                     data_type=data_type)

    def show_y_grad(self, window_name='image', kernel_size=5,
                    data_type=cv2.CV_32F):
        """Shows the Sobel's Y axis gradient of the image.

        Keyword Args:
            window_name (='image'): the name of the window in which the image
                will be displayed
            kernel_size (=5): the kernel size of the Sobel filter
            data_type (=cv2.CV_32F): the datatype to be used on the calculation
        """
        cv2.imshow(window_name, self.get_y_grad(kernel_size=kernel_size,
                                                data_type=data_type))

    def get_grayscale(self):
        """Gets the Grayscale version of the image.

        Returns:
            The Grayscale version of the image.
        """
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def apply_grayscale(self):
        """Substitutes the current image for it's Grayscale version."""
        self.image = self.get_grayscale()

    def show_grayscale(self, window_name='image'):
        """Shows the current image's Grayscale version."""
        cv2.imshow(window_name, self.get_grayscale())

    def get_complex_grad(self, kernel_size=5, data_type=cv2.CV_32F):
        """Calculates the current image's complex gradient matrix.

        This method assing the X axis gradient to the real part and Y axis
        gradient to the complex part of a matrix.

        See get_y_grad() and get_x_grad() for further documentation.

        Keyword Args:
            kernel_size (=5): the kernel size of the Sobel filter
            data_type (=cv2.CV_32F): the datatype to be used on the calculation

        Returns:
            The resulting complex Gradient matrix.
        """
        grad_x = self.get_x_grad(kernel_size=kernel_size, data_type=data_type)
        grad_y = self.get_x_grad(kernel_size=kernel_size, data_type=data_type)

        return grad_x + (0 + 1j) * grad_y

    def apply_complex_grad(self, kernel_size=5, data_type=cv2.CV_32F):
        """Substitutes the current image for it's complex gradient.

        Keyword Args:
            kernel_size (=5): the kernel size of the Sobel filter
            data_type (=cv2.CV_32F): the datatype to be used on the calculation
        """
        self.image = self.get_complex_grad(kernel_size=kernel_size,
                                           data_type=data_type)

    def lines(self):
        """Determines the number of lines of the image."""
        return self.image.shape[0]

    def cols(self):
        """Determines the number of columns of the image."""
        return self.image.shape[1]

    def copy(self):
        """Genetates a deep copy of this class instance."""
        return copy.deepcopy(self)

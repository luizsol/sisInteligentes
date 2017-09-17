# -*- coding: utf-8 -*-
"""A module to implement some basic image manipulations."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"

import copy

import cv2
import numpy as np


class Container(object):
    """A class to implement some basic image manipulations.

    Attributes:
        image (str): The image stored by the object.
    """

    def __init__(self, image=None):
        """Inits ImageProcessor with an optional image."""
        if image is not None:
            self.image = image
        else:
            self._image = None

    @property
    def image(self):
        """The image getter.

        The true image object is incapsulated on the _image attribute,
        therefore the user must use the image virtual attribute to interact
        with image stored in this class
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
        """Determines whether an image is already stored."""
        if self.image is None:
            return False
        return True

    def lines(self):
        """Determines the number of lines of the image."""
        return self.image.shape[0]

    def cols(self):
        """Determines the number of columns of the image."""
        return self.image.shape[1]

    def colors(self):
        """Determines the number of colors of the image."""
        shape = self.image.shape
        if len(shape) == 2:
            return 1
        return shape[2]

    def copy(self):
        """Genetates a deep copy of this class instance."""
        return copy.deepcopy(self)

    def show(self, window_name='image'):
        """Displays the current image.

        Keyword Args:
            window_name (='image'): the name of the window in which the image
                will be displayed
        """
        cv2.imshow(window_name, self.image)

    def save(self, path='image_processor_output.bmp'):
        """Saves the current image on the file system.

        Keyword Args:
            path (='image_processor_output'): the path where the file should be
                saved
        """
        cv2.imwrite(path, self.image)

    def get_grayscale(self):
        """Gets the Grayscale version of the image."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def apply_grayscale(self):
        """Substitutes the current image for it's Grayscale version."""
        self.image = self.get_grayscale()

    def show_grayscale(self, window_name='grayscale'):
        """Shows the current image's Grayscale version.

        Keyword Args:
            window_name (='grayscale'): the name of the window in which the
                image will be displayed
        """
        cv2.imshow(window_name, self.get_grayscale())

    def get_grad(self, kernel_size=5, data_type=cv2.CV_32F):
        """Retrieves the image's complex matrix of the storaged image.

        Keyword Args:
            kernel_size (=5): The kernel size of the Scharr filter.
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        grad_x = (1.0 / 4080) * cv2.Scharr(self.image, data_type, 1, 0,
                                           kernel_size)

        grad_y = (1.0 / 4080) * cv2.Scharr(self.image, data_type, 0, 1,
                                           kernel_size)

        return grad_x + (0 + 1j) * grad_y

    def get_grad_img(self, kernel_size=5, data_type=cv2.CV_32F):
        """Generates an image of this image's gradient.

        Keyword Args:
            kernel_size (=5): The kernel size of the Scharr filter.
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        grad = self.get_grad(kernel_size=kernel_size, data_type=data_type)

        lines = self.lines()
        cols = self.cols()
        n_grad = np.zeros((lines, cols, 3), np.uint8)

        for line in range(0, lines):
                for col in range(0, cols):
                    g_sum = np.sum(grad[line, col])
                    g_abs = np.abs(g_sum) / 3.0
                    x = np.uint8(255 * g_abs * np.cos(g_sum))
                    y = np.uint8(255 * g_abs * np.sin(g_sum))
                    n_grad[line, col, 0] = x
                    n_grad[line, col, 2] = y

        return n_grad

    def show_grad(self, kernel_size=5, data_type=cv2.CV_32F,
                  window_name='grad'):
        """Displays the current image's gradient.

        Keyword Args:
            kernel_size (=5): The kernel size of the Scharr filter.
            data_type (=cv2.CV_32F): The data type of the image elements
            window_name (='grad'): the name of the window in which the image
                will be displayed
        """
        cv2.imshow(window_name, self.get_grad(kernel_size=kernel_size,
                   data_type=data_type))

        cv2.waitKey()

    def get_gaussian_blur(self, kernel_height=5, kernel_width=5,
                          data_type=cv2.CV_32F):
        """Retrieves the result of the gaussian blur of the storaged image.

        Keyword Args:
            kernel_height (=5): The kernel height of the Gaussian filter.
            kernel_width (=5): The kernel width of the Gaussian filter.
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        return cv2.GaussianBlur(self.image, (kernel_height, kernel_width),
                                data_type)

    def apply_gaussian_blur(self, kernel_height=5, kernel_width=5,
                            data_type=cv2.CV_32F):
        """Applies the Gaussian Blur on the storaged image.

        Keyword Args:
            kernel_height (=5): The kernel height of the Gaussian filter.
            kernel_width (=5): The kernel width of the Gaussian filter.
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        self.image = self.get_gaussian_blur(kernel_height=kernel_height,
                                            kernel_width=kernel_width,
                                            data_type=data_type)

    def get_canny(self, threshold_1=100, threshold_2=200):
        """Retrieves the result of the Canny filter of the storaged image.

        Keyword Args:
            threshold_1 (=100): The lower threshold of the filter.
            threshold_2 (=200): The higher threshold of the filter.
        """
        return cv2.Canny(self.image, threshold_1, threshold_2)

    def apply_canny(self, threshold_1=100, threshold_2=200):
        """Applies the the Canny filter to the storaged image.

        Keyword Args:
            threshold_1 (=100): The lower threshold of the filter.
            threshold_2 (=200): The higher threshold of the filter.
        """
        self.image = self.get_canny(threshold_1, threshold_2)

    def show_canny(self, threshold_1=100, threshold_2=200,
                   window_name='canny'):
        """Displays the current image's Canny.

        Keyword Args:
            threshold_1 (=100): The lower threshold of the filter.
            threshold_2 (=200): The higher threshold of the filter.
            window_name (='canny'): the name of the window in which the image
                will be displayed
        """
        cv2.imshow(window_name, self.get_canny(threshold_1, threshold_2))

    def __repr__(self):
        """The representation dunder of the class."""
        if self.has_image():
            return "Container(np.array)"
        else:
            return "Container(None)"

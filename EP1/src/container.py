# -*- coding: utf-8 -*-

import copy

import cv2
import numpy as np


class Container(object):
    """docstring for Container"""

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
        cv2.waitKey()

    def save(self, path='image_processor_output.bmp'):
        """Saves the current image on the file system.

        Keyword Args:
            path (='image_processor_output'): the path where the file should be
                saved
        """
        cv2.imwrite(path, self.image)

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

    def get_grad(self, kernel_size=5, data_type=cv2.CV_32F):
        grad_x = (1.0 / 4080) * cv2.Scharr(self.image, data_type, 1, 0,
                                           kernel_size)

        grad_y = (1.0 / 4080) * cv2.Scharr(self.image, data_type, 0, 1,
                                           kernel_size)

        return grad_x + (0 + 1j) * grad_y

    def show_grad(self, kernel_size=5, data_type=cv2.CV_32F,
                  window_name='grad'):
        cv2.imshow(window_name, self.get_grad(kernel_size=kernel_size,
                   data_type=data_type))

        cv2.waitKey()

    def get_gaussian_blur(self, kernel_height=5, kernel_width=5,
                          data_type=cv2.CV_32F):
        return cv2.GaussianBlur(self.image, (kernel_height, kernel_width),
                                data_type)

    def apply_gaussian_blur(self, kernel_height=5, kernel_width=5,
                            data_type=cv2.CV_32F):
        self.image = self.get_gaussian_blur(kernel_height=kernel_height,
                                            kernel_width=kernel_width,
                                            data_type=data_type)

    def get_canny(self, threshold_1=100, threshold_2=200):
        return cv2.Canny(self.image, threshold_1, threshold_2)

    def apply_canny(self, threshold_1=100, threshold_2=200):
        self.image = self.get_canny(threshold_1, threshold_2)

    def show_canny(self, threshold_1=100, threshold_2=200,
                   window_name='canny'):
        cv2.imshow(window_name, self.get_canny(threshold_1, threshold_2))

    def __repr__(self):
        """The representation dunder of the class."""
        if self.has_image():
            return "Container(np.array)"
        else:
            return "Container(None)"

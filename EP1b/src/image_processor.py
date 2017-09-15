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

    def get_smooth_complex_grad(self, filter_kernel_size=5, grad_kernel_size=5,
                                data_type=cv2.CV_32F, iterations=3):

        old_grad = self.get_complex_grad(kernel_size=grad_kernel_size,
                                         data_type=data_type)
        new_grad = np.copy(old_grad)
        channels = self.colors()
        lines = self.lines()
        cols = self.cols()

        if channels == 1:
            old_grad.resize((lines, cols, 1), refcheck=False)
            new_grad.resize((lines, cols, 1), refcheck=False)

        for iteration in range(0, iterations):
            for line in range(filter_kernel_size // 2, lines -
                              filter_kernel_size // 2):
                for col in range(filter_kernel_size // 2, cols -
                                 filter_kernel_size // 2):
                    for channel in range(0, channels):
                        # Prototype
                        lower_line_range = line - filter_kernel_size // 2
                        upper_line_range = line + filter_kernel_size // 2
                        lower_col_range = col - filter_kernel_size // 2
                        upper_col_range = col + filter_kernel_size // 2

                        grad_sum = \
                            np.sum(old_grad[lower_line_range:upper_line_range,
                                            lower_col_range:upper_col_range,
                                            channel])

                        norm = abs(new_grad[line, col, channel])
                        real_prop = np.cos(np.angle(grad_sum))
                        im_prop = np.sin(np.angle(grad_sum))

                        new_grad[line, col, channel] = norm * real_prop + \
                            norm * (0 + 1j) * im_prop

            old_grad = np.copy(new_grad)

        if channels == 1:
            new_grad.resize((lines, cols), refcheck=False)

        return new_grad

    def get_gaussian_blur(self, kernel_height=5, kernel_width=5,
                          data_type=cv2.CV_32F):
        return cv2.GaussianBlur(self.image, (kernel_height, kernel_width),
                                data_type)

    def apply_gaussian_blur(self, kernel_height=5, kernel_width=5,
                            data_type=cv2.CV_32F):
        self.image = self.get_gaussian_blur(kernel_height=kernel_height,
                                            kernel_width=kernel_width,
                                            data_type=data_type)

    def show_gaussian_blur(self, window_name='image', kernel_height=5,
                           kernel_width=5, data_type=cv2.CV_32F):
        """Shows the Sobel's Y axis gradient of the image.

        Keyword Args:
            window_name (='image'): the name of the window in which the image
                will be displayed
            kernel_size (=5): the kernel size of the Sobel filter
            data_type (=cv2.CV_32F): the datatype to be used on the calculation
        """
        cv2.imshow(window_name, self.get_gaussian_blur(
            kernel_height=kernel_height, kernel_width=kernel_width,
            data_type=data_type))

    def get_canny(self, threshold_1, threshold_2):
        return cv2.Canny(self.image, threshold_1, threshold_2)

    def apply_canny(self, threshold_1, threshold_2):
        self.image = self.get_canny(threshold_1, threshold_2)

    def show_canny(self, threshold_1, threshold_2, window_name='image'):
        cv2.imshow(window_name, self.get_canny(threshold_1, threshold_2))

    def get_circle_hough_by_grad(self, radius, threshold=10, smooth_grad=True,
                                 smooth_grad_iterations=5):

        channels = self.colors()

        if channels == 1:
            data_type = type(self.image[0][0])
        else:
            data_type = type(self.image[0][0][0])

        result = np.zeros(self.image.shape, data_type)

        if smooth_grad:
            grad = self.get_smooth_complex_grad(
                filter_kernel_size=5, grad_kernel_size=5,
                iterations=smooth_grad_iterations)
        else:
            grad = self.get_complex_grad()

        lines = self.lines()
        cols = self.cols()

        if channels == 1:
            grad.resize((lines, cols, 1), refcheck=False)
            result.resize((lines, cols, 1), refcheck=False)

        for channel in range(0, channels):
            for line in range(0, lines):
                for col in range(0, cols):
                    value = grad[line][col][channel]
                    if abs(value) > threshold:
                        dx = int(np.cos(np.angle(value)) * radius)
                        dy = int(np.sin(np.angle(value)) * radius)
                        if ((line + dy) < lines and (line + dy) > 0) and \
                           ((col + dx) < cols and (col + dx) > 0):
                            result[line + dy, col + dx, channel] = \
                                result[line + dy, col + dx, channel] \
                                + data_type(40)

                        if ((line - dy) < lines and (line - dy) > 0) and \
                           ((col - dx) < cols and (col - dx) > 0):
                            result[line - dy, col - dx, channel] = \
                                result[line - dy, col - dx, channel] \
                                + data_type(40)
        if channels == 1:
            result.resize((lines, cols), refcheck=False)

        return result

    def show_circle_hough_by_grad(self, radius, window_name='image',
                                  threshold=10, smooth_grad=True,
                                  smooth_grad_iterations=5):
        cv2.imshow(window_name, self.get_circle_hough_by_grad(
            radius, threshold=threshold, smooth_grad=smooth_grad,
            smooth_grad_iterations=smooth_grad_iterations))

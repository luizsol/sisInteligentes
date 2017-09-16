# -*- coding: utf-8 -*-

import cv2
import numpy as np


class ImageProcessor(object):
    """A class responsible for storing images.

    Attributes:
        image: A virtual attribute to implement the setter and getter of _image
        _image: The real attribute responsible for storing the class image
    """

    def __init__(self, image=None):
        """Inits ImageProcessor with an optional image."""
        pass

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

    def get_circle_hough_by_grad(self, container, radius, threshold=10):

        channels = container.colors()

        if channels == 1:
            data_type = type(container.image[0][0])
        else:
            data_type = type(container.image[0][0][0])

        result = np.zeros(container.image.shape, data_type)

        grad = container.get_grad()

        lines = container.lines()
        cols = container.cols()

        point_value = 10

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
                                + data_type(point_value)

                        if ((line - dy) < lines and (line - dy) > 0) and \
                           ((col - dx) < cols and (col - dx) > 0):
                            result[line - dy, col - dx, channel] = \
                                result[line - dy, col - dx, channel] \
                                + data_type(point_value)

        if channels == 1:
            result.resize((lines, cols), refcheck=False)

        return result

    def show_circle_hough_by_grad(self, container, radius,
                                  window_name='complex_grad',
                                  threshold=10):
        cv2.imshow(window_name, self.get_circle_hough_by_grad(
            container, radius, threshold=threshold))
        cv2.waitKey()

    def detect_circle(self, container, radius, threshold=90,
                      data_type=cv2.CV_32F):
        points = self.get_circle_hough_by_grad(container, radius, threshold=0)
        points[points < threshold] = 0
        points = cv2.GaussianBlur(points, (9, 9), data_type)
        center = np.unravel_index(points.argmax(), points.shape)
        if points[center] > 0:
            return center
        else:
            return None

# -*- coding: utf-8 -*-
"""A module to implement some more complex image manipulations."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"


import cv2
import numpy as np


class ImageProcessor(object):
    """A class responsible for processing images."""

    def __init__(self):
        pass

    def get_circle_hough_transform_by_grad(self, container, radius,
                                           threshold=0):
        """Uses an image's gradient to determine the existence of a given
        circle in it by means of the hough transform.

        Args:
            container: a Container object containing the image to be analized.
            radius: the radius of the circle to be searched for

        Keyword Args:
            threshold (=0): The minimum absolute value of the gradient to be
                taken into consideration
        """
        channels = container.colors()

        if channels == 1:
            data_type = type(container.image[0][0])
        else:
            data_type = type(container.image[0][0][0])

        result = np.zeros(container.image.shape[:2], data_type)

        grad = container.get_grad()

        lines = container.lines()
        cols = container.cols()

        point_value = 1

        if channels == 1:
            grad.resize((lines, cols, 1), refcheck=False)

        for channel in range(0, channels):
            for line in range(0, lines):
                for col in range(0, cols):
                    value = grad[line][col][channel]
                    if abs(value) > threshold:
                        dx = int(np.cos(np.angle(value)) * radius)
                        dy = int(np.sin(np.angle(value)) * radius)
                        # Adding points on the gradient direction and by the
                        # radius distance
                        if ((line + dy) < lines and (line + dy) > 0) and \
                           ((col + dx) < cols and (col + dx) > 0):
                            result[line + dy, col + dx] += \
                                + data_type(point_value)

                        if ((line - dy) < lines and (line - dy) > 0) and \
                           ((col - dx) < cols and (col - dx) > 0):
                            result[line - dy, col - dx] += \
                                + data_type(point_value)

        return result

    def detect_circle(self, container, radius, threshold=44,
                      data_type=cv2.CV_32F):
        """Uses an image's resulting hough circle transform to determine the
        existence of a circle in it.

        Args:
            container: a Container object containing the image to be analized.
            radius: the radius of the circle to be searched for

        Keyword Args:
            threshold (=44): The minimum absolute value of a point to be
                be considered a circle center
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        points = self.get_circle_hough_transform_by_grad(
            container, radius, threshold=0)

        points = cv2.GaussianBlur(points, (5, 5), data_type)
        center = np.unravel_index(points.argmax(), points.shape)
        if points[center] > threshold:
            return center
        else:
            return None

    def detect_tyre(self, container, min_radius=50, max_radius=70,
                    show_text=True):
        """Uses an image's resulting hough circle transform to determine and
        highlight the existence of a tyre in it.

        Args:
            container: a Container object containing the image to be analized.

        Keyword Args:
            min_radius (=50): the minimum radius of the tyre to be searched for
            max_radius (=70): the maximu radius of the tyre to be searched for
            show_text (=True): whether information about the detected circle
                should be added to the image
        """
        for radius in range(min_radius, max_radius):
            center = self.detect_circle(container, radius)
            if center is not None:
                result = container.copy()
                cv2.circle(result.image, (center[1], center[0]), radius,
                           (0, 255, 255), 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if show_text:
                    cv2.putText(result.image, 'Center: ' + str(center),
                                (3, 15), font, 0.5, (255, 128, 128), 1,
                                cv2.LINE_AA)

                    cv2.putText(result.image, 'Radius: ' + str(radius),
                                (3, 30), font, 0.5, (255, 128, 128), 1,
                                cv2.LINE_AA)

                return result

        return None

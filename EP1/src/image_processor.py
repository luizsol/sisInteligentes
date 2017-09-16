# -*- coding: utf-8 -*-

import cv2
import numpy as np

from container import Container


class ImageProcessor(object):
    """A class responsible for storing images.

    Attributes:
        image: A virtual attribute to implement the setter and getter of _image
        _image: The real attribute responsible for storing the class image
    """

    def __init__(self, image=None):
        """Inits ImageProcessor with an optional image."""
        pass

    def get_circle_hough_by_grad(self, container, radius, threshold=10):

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
                        if ((line + dy) < lines and (line + dy) > 0) and \
                           ((col + dx) < cols and (col + dx) > 0):
                            result[line + dy, col + dx] += \
                                + data_type(point_value)

                        if ((line - dy) < lines and (line - dy) > 0) and \
                           ((col - dx) < cols and (col - dx) > 0):
                            result[line - dy, col - dx] += \
                                + data_type(point_value)

        return result

    def show_circle_hough_by_grad(self, container, radius,
                                  window_name='complex_grad',
                                  threshold=10):
        cv2.imshow(window_name, self.get_circle_hough_by_grad(
            container, radius, threshold=threshold))
        cv2.waitKey()

    def detect_circle(self, container, radius, threshold=44,
                      data_type=cv2.CV_32F):
        t_container = Container(container.image)
        points = self.get_circle_hough_by_grad(t_container, radius,
                                               threshold=0)

        points = cv2.GaussianBlur(points, (5, 5), data_type)
        center = np.unravel_index(points.argmax(), points.shape)
        if points[center] > threshold:
            return center
        else:
            return None

    def detect_tyre(self, container, min_radius=50, max_radius=70,
                    show_text=True):
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import cv2
import numpy as np

class ImageProcessor(object):
    """docstring for ImageProcessor"""
    def __init__(self, image=None):
        super(ImageProcessor, self).__init__()
        if image:
            self.image = image

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, other):
        if isinstance(other, np.ndarray):
            self._image = other
        elif isinstance(other, str):
            self._image = cv2.imread(other, cv2.IMREAD_UNCHANGED)
        else:
            raise TypeError

    def show_image(self, window_name=None):
        window_name = 'image' if not window_name else window_name
        cv2.imshow(window_name, self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def save_image(self, path=None):
        path = 'image_processor_output.bmp' if not path else path
        cv2.imwrite(path, self.image)

    def get_laplacian(self):
        return cv2.Laplacian(self.image, cv2.CV_64F)

    def apply_laplacian(self):
        self.image = self.get_laplacian()

    def x_grad(self, kernel_size=5):
        return cv2.Sobel(self.image, cv2.CV_64F, 1, 0, kernel_size)

    def y_grad(self, kernel_size=5):
        return cv2.Sobel(self.image, cv2.CV_64F, 0, 2, kernel_size)

    def copy(self):
        return copy.deepcopy(self)

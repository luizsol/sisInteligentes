#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from image_processor import *

if __name__ == '__main__':
    img = ImageProcessor('img/img0105.bmp')
    img.show_image(window_name="Original Image")
    img.show_laplacian(window_name="Laplacian")
    img.show_x_grad(window_name="X-Gradient")
    img.show_y_grad(window_name="Y-Gradient")

    img2 = img.copy()
    img2.apply_laplacian()
    img2.show_grayscale(window_name='Grayscale grad')

    print img.get_complex_grad()

    cv2.waitKey()
    cv2.destroyAllWindows()

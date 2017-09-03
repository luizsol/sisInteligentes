#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from image_processor import *

if __name__ == '__main__':
    # img = ImageProcessor('img/img0105.bmp')
    img = ImageProcessor('img/fillcircle.png')
    img.show_image(window_name="Original Image")
    # img.show_laplacian(window_name="Laplacian")
    # img.show_x_grad(window_name="X-Gradient")
    # img.show_y_grad(window_name="Y-Gradient")

    img.apply_gaussian_blur()
    img.apply_gaussian_blur()
    img.apply_gaussian_blur()
    img.show_canny(100, 200, window_name='Canny')
    img.apply_canny(100, 300)
    # img.show_x_grad(window_name="X-Gradient-Canny")
    # img.show_y_grad(window_name="Y-Gradient-Canny")

    img.show_circle_hough_by_grad(window_name="Circulos", radius=70, threshold=100)

    # for radius in xrange(1, 200):
    #     print 'radius = ', radius
    #     print '   max = ', img.get_circle_hough_by_grad(radius, threshold=0).max()

    cv2.waitKey()
    cv2.destroyAllWindows()

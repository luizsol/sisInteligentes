# -*- coding: utf-8 -*-

import cv2
import numpy as np

from image_processor import *
from container import *

if __name__ == '__main__':
    x = Container('../img/img0105.bmp')
    # x.apply_gaussian_blur()
    x.apply_gaussian_blur()
    x.apply_canny()
    img_p = ImageProcessor()

    results = Container()

    for radius in range(10, 90):
        results.image = img_p.get_circle_hough_by_grad(x, radius, threshold=0)
        results.save(path='../output/r' + str(radius) + '.bmp')
        center = img_p.detect_circle(x, radius)
        print(str(radius) + ': ' + str(center))
        if center is not None:
            y = Container('../img/img0105.bmp')
            cv2.circle(y.image, center, radius, (0, 0, 255), 3)
            y.save(path='../output/circle' + str(radius) + '.bmp')

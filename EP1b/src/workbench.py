# -*- coding: utf-8 -*-

import cv2
import numpy as np

from image_processor import *
from container import *

if __name__ == '__main__':
    # path = '../img/img0105.bmp'
    path = '../img/img0106.bmp'
    # path = '../img/fillcircle.png'
    x = Container(path)
    # x.apply_gaussian_blur()
    x.save(path='../output/a.bmp')
    img_p = ImageProcessor()
    print(x.lines())
    print(x.cols())

    results = Container()

    for radius in range(50, 70):
        results.image = img_p.get_circle_hough_by_grad(x, radius, threshold=0)
        results.apply_gaussian_blur()
        results.save(path='../output/r' + str(radius) + '.bmp')
        center = img_p.detect_circle(x, radius)
        print(str(radius) + ': ' + str(center))
        if center is not None:
            print(' ' + str(results.image[center]))
            y = Container(path)
            cv2.circle(y.image, (center[1], center[0]), radius, (0, 255, 255),
                       3)
            y.save(path='../output/circle' + str(radius) + '.bmp')

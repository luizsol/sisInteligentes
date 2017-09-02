#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from image_processor import *

if __name__ == '__main__':
    img1 = ImageProcessor('img/img0105.bmp')
    img.show_image(window_name="Imagem")
    img2 = img1.copy()
    img2.laplacian()

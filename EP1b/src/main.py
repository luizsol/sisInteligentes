#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from image_processor import *

if __name__ == '__main__':
    img = ImageProcessor()
    img.image = 'img/img0105.bmp'
    img.show_image(window_name="Imagem")
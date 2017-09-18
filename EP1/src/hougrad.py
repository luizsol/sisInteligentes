# -*- coding: utf-8 -*-
"""A module that implements the EP's seventh question."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"

import sys

from container import Container
from image_processor import ImageProcessor


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


img = Container(sys.argv[1])
proc = ImageProcessor()

img_name = 'Hougrad?.png'

for radius in range(int(sys.argv[-2]), int(sys.argv[-1])):
    t_im = Container(proc.get_circle_hough_transform_by_grad(img, radius))
    t_im.save(path=img_name.replace('?', str(radius)))

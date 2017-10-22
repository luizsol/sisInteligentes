# -*- coding: utf-8 -*-
"""A module that implements the EP's fourth question."""
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

if is_number(sys.argv[-1]):
    img.apply_gaussian_blur(kernel_height=2 * int(sys.argv[-1]) + 1,
                            kernel_width=2 * int(sys.argv[-1]) + 1)

img.apply_grad_img()

img.save(sys.argv[2])

# -*- coding: utf-8 -*-
"""A module that implements the EP's eighth question."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"

import sys

from container import Container
from image_processor import ImageProcessor

img = Container(sys.argv[1])
proc = ImageProcessor()

im2 = proc.detect_tyre(img, min_radius=int(sys.argv[-2]),
                       max_radius=int(sys.argv[-1]),
                       show_text=True)

if im2 is not None:
    im2.save(sys.argv[2])

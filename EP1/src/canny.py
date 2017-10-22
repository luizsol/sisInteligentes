# -*- coding: utf-8 -*-
"""A module that implements the EP's second question."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"

import sys

from container import Container

img = Container(sys.argv[1])

img.apply_canny()

img.save(sys.argv[2])

# -*- coding: utf-8 -*-
"""A module that implements the EP's first question."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"

import sys

import video_processor as vp

vp.get_frames(sys.argv[1], sys.argv[2])

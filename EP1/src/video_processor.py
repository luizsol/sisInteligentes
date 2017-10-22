# -*- coding: utf-8 -*-
"""A module to implement some basic video manipulations."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"


import cv2


def get_frames(video_path, output_folder):
    """Retrieves and saves the frames from a video.

    Args:
        video: the path to the video to be analized.
        output_folder: the path to the folder into which the resulting frames
            will be stored.

    Source: https://stackoverflow.com/questions/33311153/python-extracting- \
        and-saving-video-frames
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite("%s/quad%d.png" % (output_folder, count), image)
        count += 1

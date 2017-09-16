# -*- coding: utf-8 -*-

import cv2


def get_frames(video, output_folder):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite("%s/quad%d.png" % (output_folder, count), image)
        count += 1

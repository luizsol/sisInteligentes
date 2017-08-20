/* File: ImgProcessor.h
 *
 * Authors:
 *   Luiz Sol (luizedusol@gmail.com)
 *
 * Version: 0.1
 */

#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdint.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

class ImgProcessor{

private:
  Mat image;
  int loadFlag;

public:
  ImgProcessor();
  ImgProcessor(string path);
  ImgProcessor(Mat image);
  ~ImgProcessor();

  Mat getImage();
  void setImage(string path);
  void setImage(Mat image);

  void showImage();
  void showImage(string windowHanle);

  Mat cannyThreshold(int lowThreshold, int ratio, int kernelSize);

  Mat getEdges();
  Mat showEgdes();

};
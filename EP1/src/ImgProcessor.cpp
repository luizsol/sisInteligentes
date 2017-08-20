/* File: ImgProcessor.cpp
 *
 * Authors:
 *   Luiz Sol (luizedusol@gmail.com)
 *
 * Version: 0.1
 */

#include "ImgProcessor.h"
#include <iostream>

using namespace std;
using namespace cv;

ImgProcessor::ImgProcessor(){}

ImgProcessor::ImgProcessor(string path) :
  ImgProcessor(imread(path, CV_LOAD_IMAGE_GRAYSCALE)){}

ImgProcessor::ImgProcessor(Mat image){
  ImgProcessor::setImage(image);
}

ImgProcessor::~ImgProcessor(){}

void ImgProcessor::setImage(string path){
  ImgProcessor::setImage(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
}

void ImgProcessor::setImage(Mat image){
  this->image = image.clone();
}

Mat ImgProcessor::getImage(){
  return this->image;
}

void ImgProcessor::showImage(){
  ImgProcessor::showImage("Image");
}

void ImgProcessor::showImage(string windowHanle){
  imshow(windowHanle, ImgProcessor::getImage());
  waitKey();
}

Mat ImgProcessor::cannyThreshold(int lowThreshold, int ratio, int kernelSize){
  Mat dst, detected_edges;

  if(lowThreshold > 100){
    lowThreshold = 100;
  }

  /// Reduce noise with a kernel 3x3
  blur(ImgProcessor::getImage(), detected_edges, Size(kernelSize, kernelSize));

  /// Canny detector
  Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio,
        kernelSize);

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  ImgProcessor::getImage().copyTo( dst, detected_edges);

  return dst;
}

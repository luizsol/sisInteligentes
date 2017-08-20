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
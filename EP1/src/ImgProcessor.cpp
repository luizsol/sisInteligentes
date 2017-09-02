/* File: ImgProcessor.cpp
 *
 * Authors:
 *   Luiz Sol (luizedusol@gmail.com)
 *
 * Version: 0.1
 */

#include "ImgProcessor.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;
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

Mat_<std::complex<double>> ImgProcessor::gray_gradient(Mat image){
  Mat saida = Mat_<std::complex<double>>(image.rows, image.cols);
  for (int i = 0; i < image.rows; i++){
    for (int j = 0; j < image.cols; i++){
      m.at(r, c);
    }
  }
}

bool ImgProcessor::saveImage(string path){
  return ImgProcessor::saveImage(path, this->image);
}

bool ImgProcessor::saveImage(string path, Mat image){
  return imwrite(path, image);
}

vector<Vec3f> ImgProcessor::detectCircles(Mat image, double minDist,
                                          int minRadius, int maxRadius){
  vector<Vec3f> circles;
  HoughCircles(image, circles, CV_HOUGH_GRADIENT, 2, minDist, 100, 100,
               minRadius, maxRadius);
  return circles;
}

vector<Vec3f> ImgProcessor::detectCircles(double minDist, int minRadius,
                                          int maxRadius){
  return ImgProcessor::detectCircles(ImgProcessor::getImage(), minDist,
                                     minRadius, maxRadius);
}

void ImgProcessor::showCircles(double minDist, int minRadius, int maxRadius){
  vector<Vec3f> circles = ImgProcessor::detectCircles(minDist, minRadius,
                                                      maxRadius);
  Mat tmpImg = ImgProcessor::getImage();
  for(size_t i = 0; i < circles.size(); i++){
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         // draw the circle center
         circle( tmpImg, center, 3, Scalar(255,255,0), -1, 8, 0 );
         // draw the circle outline
         circle( tmpImg, center, radius, Scalar(255,0,255), 3, 8, 0 );
    }
    namedWindow( "circles", 1 );
    imshow( "circles", tmpImg );
    waitKey();
}
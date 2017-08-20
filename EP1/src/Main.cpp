#include <opencv2/opencv.hpp>
#include "ImgProcessor.h"
#include "iostream"

using namespace cv;
using namespace std;

/** @function main */
int main( int argc, char** argv ){
  ImgProcessor processor("img/img0105.bmp");

  imshow("canny", processor.cannyThreshold(100, 3, 3));

  ImgProcessor wheelEdges(processor.cannyThreshold(100, 3, 3));

  wheelEdges.showCircles(20, 50, 80); //Detecting the truck wheel

  /// Wait until user exit program by pressing a key

  return 0;
}
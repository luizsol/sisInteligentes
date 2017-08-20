#include <opencv2/opencv.hpp>
#include "ImgProcessor.h"
#include "iostream"

using namespace cv;
using namespace std;

/** @function main */
int main( int argc, char** argv )
{
  ImgProcessor processor("img/img0105.bmp");

  Mat canny = processor.cannyThreshold(100, 3, 3);

  imshow("canny", canny);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  }
#include "opencv2/opencv.hpp"
#include <stdint.h>
#include <iostream>
#include <string>


using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
  Mat original = imread("doge.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  Mat modified = imread("doge.jpg", CV_LOAD_IMAGE_GRAYSCALE);

  for(int r = 0; r < modified.rows; r++){
    for(int c = 0; c < modified.cols; c++){
      modified.at<uint8_t>(r, c) = modified.at<uint8_t>(r, c) * 0.5f;
    }
  }

  imshow("Original", original);
  imshow("Modified", modified);

  waitKey();

  original = imread("doge.jpg", CV_LOAD_IMAGE_COLOR);
  modified = imread("doge.jpg", CV_LOAD_IMAGE_COLOR);

  for(int r = 0; r < modified.rows; r++){
    for(int c = 0; c < modified.cols; c++){
      // BGR
      modified.at<cv::Vec3b>(r, c)[1] = modified.at<cv::Vec3b>(r, c)[1] * 0;
    }
  }

  imshow("Original", original);
  imshow("Modified", modified);

  waitKey();
  return 0;
}
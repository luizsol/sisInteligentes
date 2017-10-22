#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>


using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
  Mat file1 = imread("doge.jpg", CV_LOAD_IMAGE_UNCHANGED);
  Mat file2 = imread("doge.jpg", CV_LOAD_IMAGE_GRAYSCALE);


  namedWindow("Color", CV_WINDOW_FREERATIO);
  namedWindow("Fixed", CV_WINDOW_AUTOSIZE);

  imshow("Color", file1);
  imshow("Fixed", file2);

  resizeWindow("Color", file1.cols/2, file1.rows/2);
  resizeWindow("Fixed", file2.cols/2, file2.rows/2);

  moveWindow("Color", 100, 100);
  moveWindow("Fixed", 200 + file1.cols, 200);

  waitKey();
  return 0;
}
#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>


using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
  Mat testColor = imread("doge.jpg", CV_LOAD_IMAGE_COLOR);
  Mat testGray = imread("doge.jpg", CV_LOAD_IMAGE_GRAYSCALE);

  imshow("Color", testColor);
  imshow("Grayscale", testGray);

  waitKey();

  imwrite("graydoge.png", testGray);

  return 0;
}
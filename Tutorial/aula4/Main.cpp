#include "opencv2/opencv.hpp"
#include <stdint.h>
#include <iostream>
#include <string>


using namespace std;
using namespace cv;

int main(int argc, char const *argv[]){
  Mat original = imread("doge.jpg", CV_LOAD_IMAGE_COLOR);
  Mat splitChannels[3];

  split(original, splitChannels);

  imshow("Blue", splitChannels[0]);
  imshow("Green", splitChannels[1]);
  imshow("Red", splitChannels[2]);
  waitKey();


  splitChannels[2] = Mat::zeros(splitChannels[2].size(), CV_8UC1);

  Mat output;

  merge(splitChannels, 3, output);
  imshow("Output", output);

  waitKey();

  return 0;
}
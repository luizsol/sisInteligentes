#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]){

  string filename = "bird.avi";
  VideoCapture capture(filename);
  Mat frame;

  if( !capture.isOpened() )
      throw "Error when reading steam_avi";

  namedWindow("w", 1);
  capture.retrieve(frame);
  for( ; ; )
  {
    capture >> frame;
    if(!frame.empty()){
      imshow("w", frame);
    }
  }
  waitKey(); // key press to close window

  return 0;
}